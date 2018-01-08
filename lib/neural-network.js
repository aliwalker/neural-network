module.exports = (function() {
    /**
     * construct a neural-network.
     * 
     * @constructor
     * @public
     * @param {Number} inputSize Number of inputs to the network.
     * @param {Number} outPutSize Number of outputs of the network.
     * @param {Object} option Optional object, which can should have the following interface:
     *      interface NeuralNetOption {
     *          hiddenLayers?: HiddenLayers;
     *          hiddenLayersSize?: Array;
     *          hiddenLayersNum?: Number;
     *          learningRate?: Number;
     *          bias: Array;
     *      };
     * 
     *      interface HiddenLayers {
     *          num: Number;
     *          size: Array;
     *      };
     * 
     *      for example:
     *          { hiddenLayers: { num: 3, size: [2, 4, 3] }, learningRete: 0.3 } or
     *          { hiddenLayersNum: 3, hiddenLayersSize: [2, 5, 6], learningRate: 0.3, bias: [.35, .21, .4] }
     */
    function NeuralNet(inputSize, outputSize, option = null) {
        option = option || {};
        const hls = option.hiddenLayers || {};
        const hiddenLayersNum = hls.num || option.hiddenLayerNum || 2;
        const hiddenLayersSize = hls.size || option.hiddenLayerSize || [inputSize, inputSize];
        const model = this;
        const bias = option.bias || [];

        this.inputLayer = createLayer(inputSize, bias[0]);
        this.outputLayer = createLayer(outputSize, 0, true);
        this.hiddenLayers = new Array(hiddenLayersNum);
        this.learningRate = option.learningRate || 0.5;

        // Create hidden layers.
        if (Array.isArray(hiddenLayersSize)) {
            for (let i = 0; i < model.hiddenLayers.length; i++) {
                this.hiddenLayers[i] = createLayer(hiddenLayersSize[i], bias[i]);
            }
        } else if (typeof hiddenLayersSize === 'number') {
            for (let i = 0; i < model.hiddenLayers.length; i++) {
                this.hiddenLayers[i] = createLayer(hiddenLayersSize, bias[i]);
            }
        } else {
            throw new Error("option.hiddenLayers.size must be an array or number.");
        }

        // Set up weights between input layer & the first hidden layer randomly.
        createRandomWeights(this.inputLayer, this.hiddenLayers[0].layerSize);

        // Set up weights among hidden layers
        // & between the last hidden layer & output layer
        model.hiddenLayers.forEach((layer, i) => {
            var nextLayerNum = (i != model.hiddenLayers.length - 1) ?   // Not the last hidden layer?
                model.hiddenLayers[i + 1].layerSize :
                model.outputLayer.layerSize;

            createRandomWeights(layer, nextLayerNum);
        });
        return this;
    }

    /**
     * Create a layer with `layerSize` neurons.
     * @private
     * @param {Number} layerSize number of neurons in the layer.
     * @param {Boolean} output indicates whether it is an output layer.
     * @returns {Array} a layer with `layerSize` neurons.
     */
    function createLayer(layerSize, bias = 0, output = false) {
        const layer = {
            layerSize: layerSize,
            weights: [],
            output: [],
            bias: bias == undefined ? 0 : bias
        }
        return layer;
    }

    /**
     * Create random weights for every neuron in a layer.
     * 
     * @private
     * @param {Array} layer A layer represented by an array.
     * @param {Number} nextLayerNum Number of neurons in next layer.
     */
    function createRandomWeights(layer, nextLayerNum) {
        const weights = layer.weights;
        for (let i = 0; i < layer.layerSize; i++) {
            weights[i] = [];
            for (let j = 0; j < nextLayerNum; j++) {
                weights[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    /**
     * Used as the activation function.
     * 
     * @private
     * @param {Number} x Input to a neuron.
     * @returns {Number} the output of a neuron.
     */
    function sigmoid(x) {
        return 1.0 / ( 1.0 + Math.exp(-x) );
    }

    /**
     * Calculate the total error of the network.
     * @private
     * @param {Array} prediction   output of the network. 
     * @param {Array} actual   actual values.
     */
    function totalError(prediction, actual) {
        let error = 0;
        for (let i in prediction) {
            error += (actual[i] - prediction[i]) ** 2;
        }
        return .5 * error;
    }

    function runInput(layer, prevLayer) {
        for (let i = 0; i < layer.layerSize; i++) {
            let input = 0;
            for (let j = 0; j < prevLayer.output.length; j++) {
                input += prevLayer.output[j] * prevLayer.weights[j][i];
            }
            layer.output[i] = sigmoid(input + prevLayer.bias);
        }
    }

    /**
     * Forward pass. Used in predict and learn.
     * @param {Array} input
     */
    NeuralNet.prototype.forward = function (input) {
        const inputLayer = this.inputLayer;
        const outputLayer = this.outputLayer;
        const hiddenLayers = this.hiddenLayers;
        const hiddenLayersNum = hiddenLayers.length;

        inputLayer.output = input;
        runInput(hiddenLayers[0], inputLayer);
        for (let i = 1; i <= hiddenLayersNum - 1; i++) {
            runInput(hiddenLayers[i], hiddenLayers[i - 1]);
        }
        runInput(outputLayer, hiddenLayers[hiddenLayersNum - 1]);
    }

    /**
     * Backward pass.
     * @param {Array} target actual values used to train the neural 
     */
    NeuralNet.prototype.backward = function (target) {
        const inputLayer = this.inputLayer;
        const outputLayer = this.outputLayer;
        const hiddenLayers = this.hiddenLayers;
        const hiddenLayersNum = hiddenLayers.length;

        // Update output layer's errors and deltas.
        let output = outputLayer.output;
        outputLayer.deltas = [];
        for (let i = 0; i < output.length; i++) {
            let error = output[i] - target[i];

            outputLayer.deltas[i] = error * output[i] * (1 - output[i]);
        }

        // Update last hidden layer's errors and deltas.
        let lastIndex = hiddenLayersNum - 1;
        let weights = hiddenLayers[lastIndex].weights;
        output = hiddenLayers[lastIndex].output;
        hiddenLayers[lastIndex].deltas = [];
        let deltas = outputLayer.deltas;
        for (let i = 0; i < output.length; i++) {
            let error = 0;

            for (let j = 0; j < deltas.length; j++) {
                error += deltas[j] * weights[i][j];
                weights[i][j] -= this.learningRate * deltas[j] * output[i];
            }
            hiddenLayers[lastIndex].deltas[i] = error * output[i] * (1 - output[i]);
        }

        // Update hidden layers' errors and deltas.
        for (let i = hiddenLayersNum - 2; i >= 0; i--) {
            output = hiddenLayers[i].output;
            weights = hiddenLayers[i].weights;
            hiddenLayers[i].deltas = [];
            deltas = hiddenLayers[i + 1].deltas;
            for (let j = 0; j < output.length; j++) {
                let error = 0;

                for (let k = 0; k < deltas.length; k++) {
                    error += deltas[k] * weights[j][k];
                    weights[j][k] -= this.learningRate * deltas[k] * output[j];
                }
                hiddenLayers[i].deltas[j] = error * output[j] * (1 - output[j]);
            }
        }

        // Update input layer's errors and deltas.
        weights = inputLayer.weights;
        output = inputLayer.output;
        inputLayer.deltas = [];
        deltas = hiddenLayers[0].deltas;
        for (let i = 0; i < output.length; i++) {
            let error = 0;

            for (let j = 0; j < deltas.length; j++) {
                error += deltas[j] * weights[i][j];
                weights[i][j] -= this.learningRate * deltas[j] * output[i];
            }
            inputLayer.deltas[i] = error * output[i] * (1 - output[i]);
        }
    }


    /**************Public API****************/
    /**
     * Visilize the network in JSON.
     * @public
     * @returns a JSON representation of the network.
     */
    NeuralNet.prototype.toJson = function() {
        //console.log(this);
        return JSON.stringify(this, null, 4);
    }

    /**
     * Given an input, predicts the output of the network.
     * This method is also used in forward propagation.
     * @public
     * @param {Array} intput input values to the network.
     * @returns {Array} output values of the network. 
     */
    NeuralNet.prototype.predict = function(input) {
        if (input.length !== this.inputLayer.layerSize) {
            throw new Error('Length of input\'s array dose not match.');
        }
        var result = [];        

        this.forward(input);
        const output = this.outputLayer.output;
        for (let i = 0; i < output.length; i++) {
            result[i] = output[i];
        }
        return result;
    }

    /**
     * 
     * @param {Array} input Input to the network. 
     * @param {Array} output Actual output.
     */
    NeuralNet.prototype.learnSingle = function(input, output) {
        // Forward pass.
        this.totalError = totalError(this.predict(input), output);

        // Backward pass.
        this.backward(output);
    }

    /**
     * Use this function to train the network.
     * The train_data should have the following format:
     * [
     *   { input: [...], output: [...] },
     *   ...
     * ]
     * @param {Array} train_data 
     */
    NeuralNet.prototype.learn = function (train_data) {
        // Learn it a couple of times.
        for (let i = 0; i < 2000; i++)
            for (const example of train_data) {
                this.learnSingle(example.input, example.output);
            }
    }
    return NeuralNet;
})();
