



class spelling{
    constructor() {    
        this.theSpells = ["Accio","Aguamenti","Alohomora","Avis","Bombarda","Colovaria","Engorgio","Epoximise","Evanesco","Expelliarmus","Flipendo","Fumos","Gemino","Impedimenta","Incendio","Locomotor","Lumos","Lumos Maxima","Oppugno","Orchideous","Periculum","Reducio","Reparo","Serpensortia","Wingardium Leviosa","not a Spell"].sort() 
        this.numberOfspells = this.theSpells.length;
        this.loaded = tf.loadModel('localstorage://spelling');
    }
    predict(spellToRecognise) {
      let spellSize = 80
      let flatSp = spellToRecognise.flat(3)
      var pad = Array.apply(null, Array(spellSize - flatSp.length)).map(Number.prototype.valueOf, 0);
      return this.loaded.then(loaded => {
        return loaded.predict(tf.tensor2d(flatSp.concat(pad), [1,spellSize])).data().then(p => {
          let max = Math.max(... p)
          return {
            score: max,
            spell: this.theSpells[p.indexOf(max)],
          }
        })
      })
      throw new Error("No model loaded");
    }
    toCurve(s) {
        // max curves
        var size = 10 
        var op0 = fitCurve(s, 6);
        var op1 = fitCurve(s, 5);
        var op2 = fitCurve(s, 4);
        var op3 = fitCurve(s, 3);
        var op4 = fitCurve(s, 2);
        var op5 = fitCurve(s, 1);
        let picked
        if (op5.length <= size) {
          picked = op5;
        } else if (op4.length <= size) {
          picked = op4;
        } else if (op3.length <= size) {
          picked = op3;
        } else if (op2.length <= size) {
          picked = op2;
        } else if (op1.length <= size) {
          picked = op1;
        } else {
          picked = op0.slice(0, size);
        }
    
        return picked.map(v => v.map(va => va.map(val => {
            return Math.round(val)
        })))
     }
     normCurve(Curve) {
        const minX = Curve.reduce((a, v) => {
            return Math.min(v[0][0],v[1][0],v[2][0],v[3][0],a)
        }, 2000)
        const minY = Curve.reduce((a, v) => {
            return Math.min(v[0][1],v[1][1],v[2][1],v[3][1],a)
        }, 2000)
        const maxX = Curve.reduce((a, v) => {
            return Math.max(v[0][0],v[1][0],v[2][0],v[3][0],a)
        }, 0)
        const maxY = Curve.reduce((a, v) => {
            return Math.max(v[0][1],v[1][1],v[2][1],v[3][1],a)
        }, 0)
    
         const factor = resolution / this.distance([maxX, maxY],  [minX, minY])
         return Curve.map(va => {
             return va.map(val => [Math.floor((val[0] - minX) * factor), Math.floor((val[1] - minY) * factor)] )
         })
     }
     normSpell(theSpell) {
       
        const minX = theSpell.reduce((a, v) => {
            return Math.min(v[0],a)
        }, 2000)
        const minY = theSpell.reduce((a, v) => {
            return Math.min(v[1],a)
        }, 2000)
        const maxX = theSpell.reduce((a, v) => {
            return Math.max(v[0],a)
        }, 0)
        const maxY = theSpell.reduce((a, v) => {
            return Math.max(v[1],a)
        }, 0)
    
        const factor = resolution / this.distance([maxX, maxY],  [minX, minY])
      
        return this.normCurve(this.toCurve(spell.map(va => {
            return [Math.floor((va[0] - minX) * factor), Math.floor((va[1] - minY) * factor)]
        }).reduce((a,val,i,arr) => {
             // A-B---C drop B
             //
             // A-----B include B
             //      /
             //     C
            if (i === 0 || i + 1 === arr.length || !this.isInALine(a.slice(-1), val, arr[i+1])) { 
                a.push(val)
            }
            return a
        }, [])));
     }
     distance(pa, pb) {
         var a = pa[0] - pb[0];
         var b = pa[1] - pb[1];
         if (a || b) {
             return Math.sqrt(a*a + b*b);
         }
         return 0
     }
     isInALine(a, b, c) {
         if (a && b && c) {
             return (this.distance(a, b) + this.distance(b, c)) === this.distance(a, c)
         }
         return false
     }
     recognise(arrayOfxy) {
         return this.predict(this.normSpell(arrayOfxy))
     }
}
