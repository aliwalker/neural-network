## Neural Network
This is one of my assignment, an implementation of neural network on NodeJS.
Note that this neural's not been working perfectly. It is for a study purpose.

## Download
```
git clone git@github.com:Alieeeeen/neural-network.git
```

## Usage

Import the class from 'lib/neural-network':

```
const NeuralNet = require('lib/neural-network');
```

Create a neural network with optional configuration:

```
let inputSize = 2;
let outputSize = 1;

// This field is optional
// for example:
//   { hiddenLayers: { num: 3, size: [2, 4, 3] } } or
//   { hiddenLayersNum: 3, hiddenLayersSize: [2, 5, 6] }
let option = {
    hiddenLayers: {
        num: 3,
        size: [2, 4, 3]
    },
    learningRate: .5    // Optional learning rate.
};

let neural = new NeuralNet(inputSize, outputSize, option);
```

Then set up a train set, for example, the **XOR** train set:

```
let train_data = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
];

// train the neuralnet, the nerual will learn it a couple of times.
neural.learn(train_data);

// predict the result from an input
neural.predict([1, 1])  // should be smaller than 0.05
```
