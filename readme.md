## Neural Network
This was originally one of my assignments. We were not required to implement them on our own, though, I attempted to give it a try and this is it.


This implementation might not be very efficient, but enough for the study purpose.

## Download
```
git clone git@github.com:Alieeeeen/neural-network.git
```

## Examples

First we'll see an **XOR** function:

```js
const NeuralNet = require('lib/neural-network')    // Import the class.
const train_data = [    // Set up train data.
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
]

const neuralnet = new NeuralNet(2, 1)  // Create a neural network with first param = input size
                                        // Second param = output size.

neuralnet.learn(train_data)             // Now learn them! The network will learn it a couple of times.
console.log(neuralnet.predict([0, 0]))  // [ 0.9497415745286806 ]
```

Here's another example inspired by [node-mind](https://github.com/stevenmiller888/mind)(I've tried this example, getting that my network performs a better job on prediction):

```js
const NeuralNet = require('lib/neural-network')    // Import the class.

// Set up characters to recognize.
const a = character(
    '.#####.' +
    '#.....#' +
    '#.....#' +
    '#######' +
    '#.....#' +
    '#.....#' +
    '#.....#'
  )
  
const b = character(
  '######.' +
  '#.....#' +
  '#.....#' +
  '######.' +
  '#.....#' +
  '#.....#' +
  '######.'
)

const c = character(
  '#######' +
  '#......' +
  '#......' +
  '#......' +
  '#......' +
  '#......' +
  '#######'
)

const neuralnet = new NeuralNet(a.length, 1)    // Create a network.
const train_data = [    // Set up train data.
    { input: a, output: map('a') },
    { input: b, output: map('b') },
    { input: c, output: map('c') },
]

neuralnet.learn(train_dat)      // Now learn it a couple of times!

// Let the neural predict letter `C`. It is OK to predict it with a pixel off.
let result = neuralnet.predict(character(
  '#######' +
  '#......' +
  '#......' +
  '#......' +
  '#......' +
  '##.....' +
  '#######'
))

console.log(result)     // [ 0.5001605681771336 ]

/**
 * Map letter to a number.
 */

function map(letter) {
  if (letter === 'a') return [ 0.1 ]
  if (letter === 'b') return [ 0.3 ]
  if (letter === 'c') return [ 0.5 ]
  return 0
}

/**
 * Turn the # into 1s and . into 0s.
 */

function character(string) {
  return string
    .trim()
    .split('')
    .map(integer)

  function integer(symbol) {
    if ('#' === symbol) return 1
    if ('.' === symbol) return 0
  }
}
```

## The *option* object
The option object can be used to set up some parameters.Note that each field in this object is optional.
```js
var option = {
    hiddenLayers: {     // Used to set up hidden layers
        num: 5,         // Number of hidden layers. Set to 2 as default.
        size: [10, 9, 8, 7, 6]  // Size of each layer, this can also be a Number. Set to equal the number of inputs as default.
    },
    learningRate: .3,    // The learning rate. Set to .5 as default.
    iterations: 5000,   // Number of iterations you want the network to learn your train data. Set to 2000 as default.
    bias: [.35, .12, .28, .76]  // Bias of input & hidden layers. Set to 0 as default.
}
```

## Others
There are some other great projects you might be interested in:
- [brain.js](https://github.com/BrainJS/brain.js)
- [neuronjs](https://github.com/janhuenermann/neurojs)
