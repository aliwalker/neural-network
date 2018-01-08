const NeuralNet = require('./lib/neural-network');

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

const neural = new NeuralNet(a.length, 1);

const train_data = [
    { input: a, output: map('a') },
    { input: b, output: map('b') },
    { input: c, output: map('c') },
]

neural.learn(train_data);

let test_data = character(
    '#######' +
    '#......' +
    '#......' +
    '#......' +
    '#......' +
    '#......' +
    '#######'
)

let result = neural.predict(test_data)

console.log(result);

function character(string) {
    return string
        .trim()
        .split('')
        .map(integer)

    function integer(symbol) {
        if ('#' === symbol) return 1;
        if ('.' === symbol) return 0;
    }
}

function map(letter) {
    switch(letter) {
    case 'a':   return [ .1 ];
    case 'b':   return [ .3 ];
    case 'c':   return [ .5 ];
    default:    return 0;
    }
}
