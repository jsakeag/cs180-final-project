/* Outlier recognizer: open-vocabulary object recognition in the browser.
 *
 * Uses an open CLIP model through transformers.js. No API, no server: the model weights
 * are downloaded once from Hugging Face and cached by the browser.
 *
 *   1. On load, every object name in the vocabulary is embedded with the text encoder
 *      ("a photo of a X" + "a drawing of a X", averaged). The result is cached in localStorage,
 *      so this only happens on a player's first visit.
 *   2. Per photo, the image is embedded once and compared with every object name (dot product).
 *   3. The same image embedding is compared with every prompt, which is used to accept
 *      answers that aren't in a prompt's lists.
 *
 * To try a different model, change MODEL. Any CLIP model with ONNX weights on Hugging Face
 * that transformers.js supports will work (e.g. "Xenova/clip-vit-large-patch14" is more
 * accurate but much bigger).
 */
window.Recognizer = (() => {
  const MODEL = "Xenova/clip-vit-base-patch16";
  const LIB = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";
  const TEMPLATES = ["a photo of a {}", "a drawing of a {}"];
  const USE_WEBGPU = false;
  const D = window.OUTLIER_DATA;

  let ready = null;          // promise
  let tf, tokenizer, processor, textModel, visionModel;
  let vocab = [], vocabEmb = null, promptEmb = null, dim = 0;

  const unique = a => [...new Set(a.map(s => s.toLowerCase().trim()))];

  function buildVocab() {
    const all = [];
    for (const p of D.POOL) { all.push(...Object.keys(p.common), ...p.rare); }
    all.push(...D.GENERAL, ...D.UNCLEAR);
    return unique(all);
  }

  function normalizeRows(data, rows, cols) {
    for (let r = 0; r < rows; r++) {
      let s = 0; for (let c = 0; c < cols; c++) s += data[r*cols+c] ** 2;
      s = Math.sqrt(s) || 1; for (let c = 0; c < cols; c++) data[r*cols+c] /= s;
    }
    return data;
  }

  async function embedTexts(texts, onProgress) {
    const out = [];
    const B = 32;
    for (let i = 0; i < texts.length; i += B) {
      const inputs = tokenizer(texts.slice(i, i + B), { padding: true, truncation: true });
      const { text_embeds } = await textModel(inputs);
      dim = text_embeds.dims[1];
      out.push(new Float32Array(text_embeds.data));
      onProgress && onProgress(Math.min(1, (i + B) / texts.length));
    }
    const all = new Float32Array(texts.length * dim);
    let o = 0; for (const chunk of out) { all.set(chunk, o); o += chunk.length; }
    return normalizeRows(all, texts.length, dim);
  }

  // average the template embeddings for each label
  function poolTemplates(emb, n) {
    const T = TEMPLATES.length, res = new Float32Array(n * dim);
    for (let i = 0; i < n; i++) for (let t = 0; t < T; t++)
      for (let c = 0; c < dim; c++) res[i*dim+c] += emb[(i*T+t)*dim+c] / T;
    return normalizeRows(res, n, dim);
  }

  // ---- localStorage cache for text embeddings ----
  const cacheKey = () => "outlier-emb:" + MODEL + ":" + TEMPLATES.join("|") + ":" + hash(vocab.join(",") + D.POOL.map(p=>p.q).join(","));
  function hash(s){ let h = 5381; for (let i=0;i<s.length;i++) h = ((h<<5)+h+s.charCodeAt(i))>>>0; return h.toString(36); }
  function toB64(f32){ const b = new Uint8Array(f32.buffer); let s=""; for (let i=0;i<b.length;i+=0x8000) s += String.fromCharCode.apply(null, b.subarray(i,i+0x8000)); return btoa(s); }
  function fromB64(s){ const b = Uint8Array.from(atob(s), c => c.charCodeAt(0)); return new Float32Array(b.buffer); }
  function loadCache(){
    try { const j = JSON.parse(localStorage.getItem(cacheKey())); if (!j) return false;
      dim = j.dim; vocabEmb = fromB64(j.v); promptEmb = fromB64(j.p); return true; } catch { return false; }
  }
  function saveCache(){
    try {
      for (let i = localStorage.length - 1; i >= 0; i--) { const k = localStorage.key(i); if (k && k.startsWith("outlier-emb:")) localStorage.removeItem(k); }
      localStorage.setItem(cacheKey(), JSON.stringify({ dim, v: toB64(vocabEmb), p: toB64(promptEmb) }));
    } catch { /* storage full or blocked: we'll just recompute next visit */ }
  }

  async function loadModels(device, onProgress) {
    const files = {};
    const progress_callback = e => {
      if (e.status === "progress" && e.file) { files[e.file] = { loaded: e.loaded || 0, total: e.total || 0 }; }
      const v = Object.values(files), tot = v.reduce((a, f) => a + f.total, 0), got = v.reduce((a, f) => a + f.loaded, 0);
      const mb = n => Math.round(n / 1048576);
      const sizesKnown = v.length && v.every(f => f.total > 0);
      if (sizesKnown) onProgress(`Downloading model (${mb(got)} of ${mb(tot)} MB, one time only)`, got / tot);
      else if (got) onProgress(`Downloading model (${mb(got)} MB so far, one time only)`, 0.5);
      if (e.status === "ready") onProgress("Starting model", 1);
    };
    // Quantized (8-bit) weights: several times smaller than full precision, near-identical accuracy.
    const opts = { device, dtype: "q8", progress_callback };
    tokenizer = await tf.AutoTokenizer.from_pretrained(MODEL);
    processor = await tf.AutoProcessor.from_pretrained(MODEL);
    [textModel, visionModel] = await Promise.all([
      tf.CLIPTextModelWithProjection.from_pretrained(MODEL, opts),
      tf.CLIPVisionModelWithProjection.from_pretrained(MODEL, opts),
    ]);
  }

  function load(onProgress = () => {}) {
    if (ready) return ready;
    ready = (async () => {
      onProgress("Loading", 0);
      tf = await import(LIB);
      vocab = buildVocab();
      // WASM runs the 8-bit model everywhere; set USE_WEBGPU = true to try the GPU instead.
      const device = (USE_WEBGPU && "gpu" in navigator) ? "webgpu" : "wasm";
      try { await loadModels(device, onProgress); }
      catch (e) { if (device === "wasm") throw e; console.warn("WebGPU failed, using WASM", e); await loadModels("wasm", onProgress); }

      if (!loadCache()) {
        onProgress("Learning object names", 0);
        const texts = vocab.flatMap(v => TEMPLATES.map(t => t.replace("{}", v)));
        const raw = await embedTexts(texts, p => onProgress("Learning object names", p * 0.95));
        vocabEmb = poolTemplates(raw, vocab.length);
        promptEmb = await embedTexts(D.POOL.map(p => "a photo of " + p.q.charAt(0).toLowerCase() + p.q.slice(1)));
        saveCache();
      }
      onProgress("Ready", 1);
      return true;
    })();
    return ready;
  }

  async function classify(blob) {
    await load();
    const image = await tf.RawImage.fromBlob(blob);
    const inputs = await processor(image);
    const { image_embeds } = await visionModel(inputs);
    const img = normalizeRows(new Float32Array(image_embeds.data), 1, dim);

    const dot = (M, i) => { let s = 0; for (let c = 0; c < dim; c++) s += M[i*dim+c] * img[c]; return s; };
    const sims = vocab.map((_, i) => dot(vocabEmb, i));
    // softmax with CLIP's usual temperature (100) for readable confidences
    const mx = Math.max(...sims), ex = sims.map(s => Math.exp(100 * (s - mx))), Z = ex.reduce((a, b) => a + b, 0);
    const ranked = vocab.map((label, i) => ({ label, p: ex[i] / Z })).sort((a, b) => b.p - a.p).slice(0, 8);
    const promptScores = D.POOL.map((_, i) => dot(promptEmb, i));
    return { ranked, promptScores };
  }

  /* Does this label count as an answer to prompt #pi?
     1. Known valid (in the prompt's common or rare list) → yes.
     2. Otherwise, the photo itself must look more like this prompt than most other prompts
        (top 3 of all prompts by image-prompt similarity). Cheap and uncalibrated: tune with real photos. */
  function fits(pi, label, promptScores) {
    if (D.UNCLEAR.includes(label)) return false;
    const p = D.POOL[pi];
    if (label in p.common || p.rare.includes(label)) return true;
    if (!promptScores) return false;
    const mine = promptScores[pi];
    return promptScores.filter(s => s > mine).length < 3;
  }

  return { load, classify, fits, MODEL };
})();
