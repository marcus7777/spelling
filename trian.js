import gestureSpells from './lib/gesture-spells.js';
  //import * as tf from '@tensorflow/tfjs';

export function setup() {
  if (!Array.prototype.flat) {
    Array.prototype.flat = function() {
      return this.reduce(function (flat, toFlatten) {
        return flat.concat(Array.isArray(toFlatten) ? toFlatten.flat() : toFlatten);
      }, []);
    }
  }
  let gr = new gestureSpells();
  let c = document.querySelector('canvas');
  let csv = document.querySelector('textarea');
  let addTraining = document.querySelector('input#addTraining');
  let cx = c.getContext('2d');
  let mousedown = false;
  let oldX = null;
  var oldY = null;
  let resolution = 27;
  let spell = [];
  let spells = [];
  let spellsOnLine;
 
  let labelsCurves = {};
  for (let index = 0; index < gr.theSpells.length; ++index) {
    if (localStorage.getItem("labels")) {
      labelsCurves[gr.theSpells[index]] = JSON.parse(localStorage.getItem("labels"))[gr.theSpells[index]] || [];
    } else {
      labelsCurves[gr.theSpells[index]] = [];
    }
  }
  function bump(shape) {
    if (shape) {
      var angle = 20;
      var radians = ((Math.PI / 180) * angle * (Math.random()-.5))
      var cos = Math.cos(radians)
      var sin = Math.sin(radians)
      var c = resolution / 2
      return shape.map(arra => arra.map((xy) => { 
        const nx = (cos * (xy[0] - c)) + (sin * (xy[1] - c)) + c;
        const ny = (cos * (xy[1] - c)) - (sin * (xy[0] - c)) + c;
        return [nx, ny];
      }));
    }
  }
  function addToArray(x, y) {
    if (x > 0 && y > 0) {
      spell.push([x, y])
    }
  }
  document.getElementById("SaveModal").addEventListener("click", function() {
    tf.io.copyModel('localstorage://spelling', 'downloads://spelling');
  });
  var x = document.getElementById("select");
  gr.theSpells.map(t => {
    var option = document.createElement("option");
    option.text = t;
    x.add(option);
  });
  function getRandom(arr) {
    return arr[Math.floor(Math.random()*arr.length)]
  }
  function unround(val) {
    return val + ((window.crypto.getRandomValues( new Uint8Array(1) )[0]/254)-.5 +(window.crypto.getRandomValues( new Uint8Array(1) )[0]/254)/10-.05) // Math.random()-.5
  } 
  document.getElementById("Train").addEventListener("click", async function() {
    let labelsCurves = {};
    for (let index = 0; index < gr.theSpells.length; ++index) {
      if (localStorage.getItem("labels")) {
        labelsCurves[gr.theSpells[index]] = JSON.parse(localStorage.getItem("labels"))[gr.theSpells[index]] || [];
      } else {
        labelsCurves[gr.theSpells[index]] = [];
      }
    }
    // findShape
    const longestSpell = 10
    const spellSize = 2 * 4 * longestSpell;
    document.getElementById("Train").disabled = true;
    document.getElementById("Retrain").disabled = true;
    var Looping = 0
    function getRandomSpell() {
      var label  = getRandom(Object.keys(labelsCurves));
      var aSpell = bump(getRandom(labelsCurves[label]));
      if (aSpell) {
        var theUnroundSpell = aSpell.map(curve => curve.map(xy => xy.map(val => {
          let unrounded = unround(val);
          if (unrounded < 0) {
            return 0;
          }
          if (unrounded > resolution) {
            return resolution;
          }
          return unrounded;
        })));
        Looping = 0
      } else {
        Looping++
        console.log(Looping)
        return getRandomSpell()
      }
 
      let arrayOfZeroBut1 = []
      for (let i = 0; i < gr.numberOfspells; i++) {
        if (Object.keys(labelsCurves).indexOf(label) === i) {
          arrayOfZeroBut1.push(1)
        } else {
          arrayOfZeroBut1.push(0)
        } 
      }
        
      return [ arrayOfZeroBut1, theUnroundSpell.flat(3)]
    }
  
    function getBatch(batchSize) {
      let batchSpellsArray = []
      let batchLabelsArray = []
 
      for (let i = 0; i < batchSize; i++) {
        var spell = getRandomSpell();
        batchSpellsArray.push(spell[1]);
        var pad = Array.apply(null, Array(spellSize - spell[1].length)).map(Number.prototype.valueOf, 0)
        batchSpellsArray.push(pad);
        batchLabelsArray.push(spell[0]);
      }
  
      const xs     = tf.tensor2d(batchSpellsArray.flat(2), [batchSize, spellSize] ); //,      'bool');
      const labels = tf.tensor2d(batchLabelsArray, [batchSize, gr.numberOfspells] );    //,      'bool');
 
      return {xs, labels};
    }
     
    const model = tf.sequential();
 
    const BATCH_SIZE = 80;
    const TRAIN_BATCHES = 400000;
    const LEARNING_RATE = 0.00001;
    const optimizer = tf.train.adam(LEARNING_RATE);
    model.add(tf.layers.dense({units: 80,inputShape: [2 * 4 * longestSpell], kernelInitializer: 'varianceScaling', activation: 'elu'}));
    model.add(tf.layers.dense({units: 80, kernelInitializer: 'varianceScaling', activation: 'relu'}));
    model.add(tf.layers.dense({units: 80, kernelInitializer: 'varianceScaling', activation: 'relu'}));
    model.add(tf.layers.dense({units: gr.numberOfspells, kernelInitializer: 'varianceScaling', activation: 'softmax'}));
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
 
    // Every few batches, test accuracy over many examples. Ideally, we'd compute
    // accuracy over the whole test set, but for performance we'll use a subset.
    const TEST_BATCH_SIZE = 1;
    const TEST_ITERATION_FREQUENCY = 5;
    const SAVE_ITERATION_FREQUENCY = 2000;
    for (let i = 0; i < TRAIN_BATCHES; i++) {
      const batch = getBatch(BATCH_SIZE);
  
      await tf.nextFrame();
      let testBatch;
      let validationData;
      // Every few batches test the accuracy of the mode.
      if (i % TEST_ITERATION_FREQUENCY === 0) {
        testBatch = getBatch(TEST_BATCH_SIZE);
        validationData = [
          testBatch.xs, testBatch.labels
        ];
      }
  
      // The entire dataset doesn't fit into memory so we call fit repeatedly
      // with batches.
      const history = await model.fit( batch.xs, batch.labels, {batchSize: BATCH_SIZE, validationData, epochs: 1});
  
      await tf.nextFrame();
      batch.xs.dispose();
      batch.labels.dispose();
      if (testBatch != null) {
        console.log("% Done/Accuracy " + Math.round(i/TRAIN_BATCHES*1000)/10, Math.round( history.history.acc[0] * 100000)/1000);
        testBatch.xs.dispose();
        testBatch.labels.dispose();
      }
      if (i && i % SAVE_ITERATION_FREQUENCY === 0 && history.history.acc[0] > .99999) {
         break;
      }
      await tf.nextFrame();
    }
    model.save('localstorage://spelling').then(v => {
      console.log("saved", v);
      gr.loadModel("localstorage");
      document.getElementById("Train").disabled = false;
      document.getElementById("Retrain").disabled = false;
    });
  })
  document.getElementById("Retrain").addEventListener("click", async function() {
    let labelsCurves = {};
    for (let index = 0; index < gr.theSpells.length; ++index) {
      if (localStorage.getItem("labels")) {
        labelsCurves[gr.theSpells[index]] = JSON.parse(localStorage.getItem("labels"))[gr.theSpells[index]] || [];
      } else {
        labelsCurves[gr.theSpells[index]] = [];
      }
    }
    // findShape
    const longestSpell = 10
    const spellSize = 2 * 4 * longestSpell;
    document.getElementById("Train").disabled = true;
    document.getElementById("Retrain").disabled = true;
    var Looping = 0
    function getRandomSpell() {
      var label  = getRandom(Object.keys(labelsCurves));
      var aSpell = bump(getRandom(labelsCurves[label]));
      if (aSpell) {
        var theUnroundSpell = aSpell.map(curve => curve.map(xy => xy.map(val => {
          let unrounded = unround(val);
          if (unrounded < 0) {
            return 0;
          }
          if (unrounded > resolution) {
            return resolution;
          }
          return unrounded;
        })));
        Looping = 0
      } else {
        Looping++
        console.log(Looping)
        return getRandomSpell()
      }
 
      let arrayOfZeroBut1 = []
      for (let i = 0; i < gr.numberOfspells; i++) {
          if (Object.keys(labelsCurves).indexOf(label) === i) {
            arrayOfZeroBut1.push(1)
          } else {
            arrayOfZeroBut1.push(0)
          } 
      }
        
      return [ arrayOfZeroBut1, theUnroundSpell.flat(3)]
    }
  
    function getBatch(batchSize) {
      let batchSpellsArray = []
      let batchLabelsArray = []
 
      for (let i = 0; i < batchSize; i++) {
        var spell = getRandomSpell();
        batchSpellsArray.push(spell[1]);
        var pad = Array.apply(null, Array(spellSize - spell[1].length)).map(Number.prototype.valueOf, 0)
        batchSpellsArray.push(pad);
        batchLabelsArray.push(spell[0]);
      }
  
      const xs     = tf.tensor2d(batchSpellsArray.flat(2), [batchSize, spellSize] ); //,      'bool');
      const labels = tf.tensor2d(batchLabelsArray, [batchSize, gr.numberOfspells] );    //,      'bool');
 
      return {xs, labels};
    }

    let model;
    try { 
       model = await tf.loadModel('localstorage://spelling');
    } catch(e) {
       model = await tf.loadModel('spelling.json');
    }
    const BATCH_SIZE = 80;
    const TRAIN_BATCHES = 400000;
    const LEARNING_RATE = 0.00001;
    const optimizer = tf.train.adam(LEARNING_RATE);
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
 
    // Every few batches, test accuracy over many examples. Ideally, we'd compute
    // accuracy over the whole test set, but for performance we'll use a subset.
    const TEST_BATCH_SIZE = 5;
    const TEST_ITERATION_FREQUENCY = 50;
    const SAVE_ITERATION_FREQUENCY = 2000;
    for (let i = 0; i < TRAIN_BATCHES; i++) {
      const batch = getBatch(BATCH_SIZE);
  
      await tf.nextFrame();
      let testBatch;
      let validationData;
      // Every few batches test the accuracy of the mode.
      if (i % TEST_ITERATION_FREQUENCY === 0) {
        testBatch = getBatch(TEST_BATCH_SIZE);
        validationData = [
          testBatch.xs, testBatch.labels
        ];
      }
  
      // The entire dataset doesn't fit into memory so we call fit repeatedly
      // with batches.
      const history = await model.fit( batch.xs, batch.labels, {batchSize: BATCH_SIZE, validationData, epochs: 1});
  
      await tf.nextFrame();
      batch.xs.dispose();
      batch.labels.dispose();
      if (testBatch != null) {
        console.log("Accuracy " + history.history.acc[0]);
        testBatch.xs.dispose();
        testBatch.labels.dispose();
      }
      if (i && i % SAVE_ITERATION_FREQUENCY === 0 && history.history.acc[0] > .99999) {
         break;
      }
      await tf.nextFrame();
    }
    model.save('localstorage://spelling').then(v => {
      console.log("saved", v);
      gr.loadModel("localstorage");
      document.getElementById("Train").disabled = false;
      document.getElementById("Retrain").disabled = false;
    });
  })
  
  function setupCanvas() {
    const e = document.getElementById("select");
    spells = labelsCurves[e.options[e.selectedIndex].text] || [];
    c.height = window.innerHeight - 30;
    c.width = window.innerWidth;
    cx.lineWidth = 10;
    cx.lineCap = 'round';
    cx.strokeStyle = 'rgb(0, 0, 50)';
    oldX = null;
    oldY = null;
    spells.map((s, idx) => {
      oldX = null;
      oldY = null;
      spellsOnLine = Math.floor(c.width / 30) - 1;
      const addY = (Math.floor(idx / spellsOnLine) * 30) + 10;
      const addX = 15 + ((idx % spellsOnLine) * 30);
      oldX = null;
      oldY = null;

      bump(s).map((p, i, spell) => {
        cx.beginPath();
        cx.moveTo(p[0][0] + addX, p[0][1] + addY);
        cx.bezierCurveTo(p[1][0] + addX, p[1][1] + addY,p[2][0] + addX, p[2][1] + addY, p[3][0] + addX, p[3][1] + addY);
        cx.strokeStyle = 'rgb(128, 128, 128)';
        cx.lineWidth = 3;
        cx.stroke();
        cx.strokeStyle = 'rgb(0, 0, 0)';
        cx.lineWidth = 1;
        cx.stroke();
        cx.closePath();
        if (!i) {
          cx.beginPath();
          cx.moveTo(p[0][0] + addX, p[0][1] + addY);
          cx.lineTo(p[0][0] + addX, p[0][1] + addY);
          cx.strokeStyle = 'rgb(0,0,0)';
          cx.lineWidth = 5;
          cx.stroke();
          cx.strokeStyle = 'rgb(175, 255, 175)';
          cx.lineWidth = 4;
          cx.stroke();
          cx.closePath();
        }
        if (i + 1 == spell.length) {
          cx.beginPath();
          cx.moveTo(p[3][0] + addX, p[3][1] + addY);
          cx.lineTo(p[3][0] + addX, p[3][1] + addY);
          cx.strokeStyle = 'rgb(0,0,0)';
          cx.lineWidth = 5;
          cx.stroke();
          cx.strokeStyle = 'rgb(255, 175, 175)';
          cx.lineWidth = 4;
          cx.stroke();
          cx.closePath();
        } 
        
      })
      cx.strokeStyle = 'rgb(0, 0, 50)';
      cx.lineWidth = 10;
    })
  }
  function onmousedown(ev) {
    mousedown = true;
    ev.preventDefault();
  }
  function onmouseup(ev) {
    mousedown = false;
    ev.preventDefault();
    let index
    if (spell.length > 5) {
      gr.recognise(spell).then(p => {
          if (p.score > .85) {
              csv.value = p.spell + " score:" + p.score
          } else {
              csv.value = "?"
          }
      });
      save(gr.normSpell(spell));
    } else if (addTraining.checked === true) {
      if (ev.changedTouches) {
        index = (Math.floor((ev.changedTouches["0"].clientY - 10) / 30) * spellsOnLine) + Math.floor((ev.changedTouches["0"].clientX - 15) / 30)
      } else {
        index = (Math.floor((ev.clientY - 10) / 30) * spellsOnLine) + Math.floor((ev.clientX - 15) / 30)
      }
      spells.splice(index, 1)
      oldX = null
      oldY = null
      spell = [];
      const e = document.getElementById("select");
      labelsCurves[e.options[e.selectedIndex].text] = spells;
      localStorage.setItem("labels", JSON.stringify(labelsCurves))
    }
    // reset
    setupCanvas();
  }
  var uniqueArray = function(arrArg) {
    return arrArg.filter(function(elem, pos,arr) {
      return arr.map(v => JSON.stringify(v)).indexOf(JSON.stringify(elem)) == pos;
    });
  };
  function save(spellToSave) {
    oldX = null
    oldY = null
    if (addTraining.checked === true) {
      spells.push(spellToSave)
      const e = document.getElementById("select");
      labelsCurves[e.options[e.selectedIndex].text] = uniqueArray(spells);
      localStorage.setItem("labels", JSON.stringify(labelsCurves))
      setupCanvas();
    }
    spell = []
  }
  const selectBox = document.getElementById("select")
  selectBox.addEventListener("change", function() {
    spells = labelsCurves[selectBox.options[selectBox.selectedIndex].text] || [];
    setupCanvas()
  })
  let lastPoint = performance.now();
  function onmousemove(ev) {
    if (lastPoint + 40 < performance.now()) { // limit to 25
      var x 
      var y  
      if (ev.changedTouches) { 
        x = ev.changedTouches["0"].clientX;
        y = ev.changedTouches["0"].clientY;  
        mousedown = true;
      } else {
        x = ev.clientX;
        y = ev.clientY;  
      }
      if (mousedown) {
        paint(x, y);
        addToArray(x, y);
      } 
      lastPoint = performance.now();
    }
  }
  function paint(x, y) {
    cx.beginPath();
    if (oldX > 0 && oldY > 0) {
      cx.moveTo(oldX, oldY);
    }
    cx.lineTo(x, y);
    cx.stroke();
    cx.closePath();
    oldX = x;
    oldY = y;
  }
  
  c.addEventListener('mousedown', onmousedown, false);
  c.addEventListener('mouseup', onmouseup, false);
  c.addEventListener('mousemove', onmousemove, false);
  c.addEventListener("touchstart", onmousedown, false);
  c.addEventListener("touchend", onmouseup, false);
  c.addEventListener("touchcancel", onmouseup, false);
  c.addEventListener("touchmove", onmousemove, false);

  setupCanvas();
}
