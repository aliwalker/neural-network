const NeuralNet = require('./lib/neural-network');

const neural = new NeuralNet(2, 1, {
    learningRate: .5
});

let train_data = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] }
]

neural.learn(train_data);

console.log(neural.toJson())
console.log(neural.predict([1, 1]));
