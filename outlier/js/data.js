/* Outlier game data. Everything here is precomputed: no model is called to score rarity.
 *
 * Each prompt has:
 *   q      the prompt shown to the player
 *   common share of typical answers (rough estimates; replace with real data when you have it)
 *   rare   valid answers that are uncommon but known to fit (scored as rare)
 * An answer the recognizer finds that is in neither list can still count if the photo
 * matches the prompt (see recognizer.js), and then it scores as an outlier.
 */
window.OUTLIER_DATA = (() => {
  const POOL = [
    { q: "Something with a hinge",
      common: {door:.29,laptop:.22,cabinet:.12,"jewelry box":.07,eyeglasses:.06,"toilet seat":.045,gate:.035,"compact mirror":.03,"car door":.025,notebook:.015},
      rare: ["clam","oyster","mussel","piano","locket","waffle iron","mailbox","toolbox","pocket watch","garlic press","nutcracker","stapler","tongs","refrigerator","oven","dishwasher","suitcase","lunchbox","flip phone","guitar case"] },
    { q: "Something mostly red",
      common: {apple:.25,tomato:.15,strawberry:.12,"stop sign":.06,"fire extinguisher":.05,rose:.05,car:.045,lipstick:.035,"chili pepper":.03,cherry:.03,mug:.025},
      rare: ["ladybug","cardinal","fire truck","lobster","radish","pomegranate","raspberry","red bell pepper","brick","fire hydrant","crab","postbox","lantern","red balloon","boxing glove"] },
    { q: "Something with wheels",
      common: {car:.30,bicycle:.21,skateboard:.09,"office chair":.08,suitcase:.07,"shopping cart":.05,stroller:.04,"toy car":.04,wheelbarrow:.02},
      rare: ["roller skate","wheelchair","tractor","train","airplane","scooter","wagon","lawn mower","hand truck","unicycle","pizza cutter","tricycle","go-kart","computer mouse","rolling pin"] },
    { q: "Something that holds water",
      common: {"water bottle":.29,cup:.21,glass:.16,mug:.11,bowl:.08,bucket:.05,vase:.045,kettle:.04,"watering can":.025},
      rare: ["canteen","thermos","aquarium","bathtub","sink","teapot","pitcher","water balloon","coconut","flask","fish bowl","test tube","beaker","hot water bottle","kiddie pool"] },
    { q: "Something with teeth that isn't an animal",
      common: {comb:.32,zipper:.19,saw:.12,fork:.11,key:.10,gear:.055,"hair clip":.035,"tape dispenser":.02},
      rare: ["rake","hacksaw","chainsaw","pliers","bottle cap","bicycle chain","zip tie","cheese grater","sprocket","staple remover","bread knife","pinking shears","wrench"] },
    { q: "Something you wear on your head",
      common: {hat:.28,"baseball cap":.21,headphones:.11,beanie:.10,helmet:.08,sunglasses:.05,headband:.04,crown:.02},
      rare: ["tiara","turban","bandana","swim cap","wig","earmuffs","beret","sombrero","shower cap","party hat","goggles","headlamp","flower crown","visor","cowboy hat"] },
    { q: "Something that makes a sound",
      common: {phone:.27,guitar:.17,speaker:.16,"alarm clock":.12,bell:.08,keys:.06,whistle:.05,piano:.05,headphones:.035},
      rare: ["kazoo","harmonica","drum","rattle","wind chime","tambourine","maraca","squeaky toy","doorbell","music box","bubble wrap","xylophone","violin","trumpet","typewriter"] },
    { q: "Something with a spiral",
      common: {"spiral notebook":.31,spring:.19,"snail shell":.14,seashell:.10,corkscrew:.09,lollipop:.065,fusilli:.05,"phone cord":.025},
      rare: ["slinky","drill bit","screw","cinnamon roll","fern","nautilus shell","pine cone","sunflower","snail","spiral staircase","bedspring","fossil","hose reel"] },
    { q: "Something soft",
      common: {pillow:.30,blanket:.18,"teddy bear":.16,towel:.11,sweater:.07,sock:.05,cat:.05,sponge:.03,"cotton ball":.025},
      rare: ["marshmallow","plush toy","feather","slipper","scarf","bread","peach","beanbag","moss","dough","cushion","earplug","rabbit","mitten","yarn","pom pom"] },
    { q: "Something with buttons",
      common: {"remote control":.28,keyboard:.19,phone:.12,shirt:.11,calculator:.08,"game controller":.08,coat:.05,microwave:.03},
      rare: ["doorbell","camera","radio","blender","washing machine","cash register","smartwatch","cardigan","flashlight","stopwatch","walkie-talkie","typewriter","accordion","elevator panel","jukebox"] },
    { q: "Something with a hole through it",
      common: {donut:.29,bagel:.19,ring:.14,button:.11,"key ring":.09,cd:.06,needle:.05,washer:.04,cheerio:.025},
      rare: ["bead","lifesaver candy","pretzel","colander","straw","pipe","flute","hex nut","penne","tire","hula hoop","embroidery hoop","spool","bracelet","pineapple ring"] },
    { q: "Something you pour from",
      common: {bottle:.28,kettle:.17,pitcher:.14,teapot:.12,"milk carton":.08,"watering can":.05,"measuring cup":.04,"gravy boat":.016},
      rare: ["ladle","jug","decanter","funnel","cocktail shaker","soap dispenser","syrup bottle","ketchup bottle","oil can","french press","coffee pot","wine bottle","beaker","creamer"] },
    { q: "Something made of wood",
      common: {chair:.23,table:.18,pencil:.15,"wooden spoon":.12,"cutting board":.09,chopsticks:.06,door:.05,guitar:.03,clothespin:.025},
      rare: ["ruler","baseball bat","barrel","birdhouse","toothpick","matchstick","popsicle stick","rolling pin","picture frame","violin","clothes hanger","drumstick","wooden bowl","chess piece","cork"] },
    { q: "Something with a cord",
      common: {"phone charger":.29,headphones:.19,lamp:.14,"hair dryer":.11,"laptop charger":.09,kettle:.05,hoodie:.04,blinds:.02},
      rare: ["toaster","clothes iron","vacuum cleaner","telephone","yo-yo","kite","tea bag","blender","fan","extension cord","microphone","jump rope","shoelace","lanyard","string lights","curling iron"] },
    { q: "Something see-through",
      common: {glass:.29,"water bottle":.23,window:.15,jar:.11,"plastic bag":.08,eyeglasses:.07,"plastic wrap":.035,"ice cube":.03},
      rare: ["magnifying glass","bubble","light bulb","marble","crystal","prism","contact lens","snow globe","glass vase","plastic cup","test tube","sandwich bag","fish bowl","goggles","jellyfish"] },
    { q: "Something round and flat",
      common: {plate:.30,coin:.24,cd:.10,frisbee:.08,coaster:.08,cookie:.065,pizza:.055,"vinyl record":.045,pancake:.02},
      rare: ["tortilla","hockey puck","dartboard","clock","mirror","button","jar lid","poker chip","cracker","sticker","cymbal","tambourine","manhole cover","lily pad","sand dollar","pepperoni"] },
    { q: "Something with a lid",
      common: {jar:.29,pot:.21,box:.15,"trash can":.11,"water bottle":.09,laptop:.06,cooler:.035,toilet:.03},
      rare: ["kettle","piano","teapot","treasure chest","butter dish","sugar bowl","lunchbox","pill bottle","paint can","toolbox","washing machine","music box","mailbox","slow cooker","rice cooker","cookie jar"] },
    { q: "Something that's growing",
      common: {plant:.33,"potted plant":.17,tree:.13,flower:.12,cactus:.08,succulent:.07,basil:.045,grass:.045},
      rare: ["mushroom","moss","sprout","bonsai","mint","tomato plant","seedling","bamboo","ivy","fern","orchid","dandelion","lichen","puppy","kitten","sourdough starter"] },
    { q: "Something with a face that isn't a person",
      common: {dog:.25,cat:.23,clock:.17,"teddy bear":.11,watch:.09,doll:.07,"electrical outlet":.035,coin:.03},
      rare: ["jack-o'-lantern","coconut","owl","fish","frog","smiley face sticker","mask","statue","puppet","snowman","figurine","action figure","robot","bust","emoji pillow"] },
    { q: "Something sticky",
      common: {tape:.33,"sticky note":.20,honey:.14,"glue stick":.12,"band-aid":.08,"lint roller":.05,stamp:.035,velcro:.025},
      rare: ["slime","chewing gum","syrup","jam","lollipop","duct tape","flypaper","glue","sticker","peanut butter","caramel","price tag","tree sap","spider web","gecko toy"] },
    { q: "Something smaller than a coin",
      common: {button:.25,"paper clip":.19,bead:.12,earring:.10,"coin battery":.09,seed:.075,pea:.06,thumbtack:.05,"rice grain":.025},
      rare: ["sesame seed","ant","ladybug","grain of sand","sim card","pill","staple","pin","screw","nail","lego piece","die","pearl","peppercorn","lentil","sprinkle"] },
  ];

  // Extra everyday objects so the recognizer can name things that fit no prompt list.
  const GENERAL = ["book","laptop","phone","cup","bottle","chair","table","bag","backpack","shoe","sneaker","sock","shirt","pants","jacket",
    "lamp","pen","pencil","scissors","knife","spoon","fork","plate","bowl","banana","orange","lemon","carrot","potato","onion","egg",
    "bread","sandwich","pizza","cake","remote control","television","monitor","keyboard","computer mouse","headphones","wallet","watch",
    "glasses","umbrella","toothbrush","toothpaste","soap","towel","pillow","blanket","bed","couch","rug","plant","flower","vase","candle",
    "clock","mirror","picture frame","box","basket","toy","ball","teddy bear","doll","car","bicycle","dog","cat","bird","tree","rock",
    "leaf","stick","shell","coin","key","ring","necklace","hat","glove","belt","paper","envelope","notebook","calculator","charger",
    "battery","light bulb","fan","heater","pot","pan","kettle","toaster","microwave","refrigerator","sink","faucet","bucket","broom",
    "trash can","door","window","stairs","guitar","drum","ball of yarn","sticker","magnet","tape measure","hammer","screwdriver","wrench"];

  // Labels that mean "no clear object": scored as unclear.
  const UNCLEAR = ["person","human face","hand","blank wall","floor","ceiling","sky","text document","screenshot","blurry photo","dark photo"];

  return { POOL, GENERAL, UNCLEAR, ROUNDS: 3, LIMIT: 180, UNLISTED: 0.002, RARE_SHARE: 0.006 };
})();
