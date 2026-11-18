# Outlier

Snap the least obvious thing that fits. 3 prompts, 3 minutes each. Obvious answers score low.

A static web app: no backend, no API keys. Object recognition runs in the player's browser.

## Run it

The page has to be served over http (opening `index.html` straight from Finder can block the model download).

```bash
cd ~/Downloads/outlier
python3 -m http.server 8000
# open http://localhost:8000
```

To put it online, upload the folder to any static host (Netlify, Vercel, GitHub Pages, Cloudflare Pages, S3).
Camera capture on phones needs HTTPS, which all of those provide.

## How it works

| Step | Where | Cost per play |
|---|---|---|
| Recognize the object | `js/recognizer.js`: open CLIP model (`Xenova/clip-vit-base-patch16`) via [transformers.js](https://huggingface.co/docs/transformers.js), in the browser (WebGPU when available, otherwise WASM) | Free (player's device) |
| Does it fit the prompt? | Known answer lists in `js/data.js`; otherwise the photo must match this prompt better than most other prompts | Free |
| How rare is it? | Precomputed answer shares in `js/data.js`, scored as surprisal (−log₂ share), scaled so an unlisted answer = 100 | Free |

First visit: the browser downloads the 8-bit model weights once (the loading bar shows the size; cached afterwards) and embeds the ~700 object
names once (cached in localStorage). After that, each photo is one image embedding plus a few hundred dot products.

If the recognizer can't load, the game still works: players type what their object is.
If the recognizer gets it wrong, players can pick from its other guesses ("Not it?") or type the name.

## Files

```
index.html          markup
css/style.css       styles
js/data.js          prompts, answer shares, rare-but-valid answers, extra vocabulary
js/recognizer.js    CLIP loading, caching, classification, fit check
js/app.js           game flow, timer, scoring, screens
favicon.svg
```

## Tuning

- **Prompts and rarity:** edit `POOL` in `js/data.js`. `common` shares are estimates. Replace them with real answer counts once you have players.
- **Rounds / time:** `ROUNDS` and `LIMIT` (seconds) at the bottom of `js/data.js`.
- **Model:** change `MODEL` in `js/recognizer.js`. `Xenova/clip-vit-large-patch14` is more accurate and much heavier.
  SigLIP 2 / MobileCLIP2 need a swap to the matching transformers.js model classes.
- **Fit check:** `fits()` in `js/recognizer.js` uses "top 3 of all prompts" for answers not in the lists. It isn't calibrated:
  test with real photos and adjust.

## Known limits

- Recognition is limited to the vocabulary (all answers in `data.js` plus `GENERAL`). Add names to recognize more things.
- Players' photos never leave their device.
