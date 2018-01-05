const random = require('random-distrib.js')

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
     *      };
     * 
     *      interface HiddenLayers {
     *          num: Number;
     *          size: Array;
     *      };
     * 
     *      for example:
     *          { hiddenLayers: { num: 3, size: [2, 4, 3] } } or
     *          { hiddenLayersNum: 3, hiddenLayersSize: [2, 5, 6] }
     */
    function NeuralNet(inputSize, outputSize, option = null) {
        option = option || {};
        var hls = option.hiddenLayers || {};
        var hiddenLayersNum = hls.num || option.hiddenLayerNum || 2;
        var hiddenLayersSize = hls.size || option.hiddenLayerSize || [inputSize, inputSize];
        var model = this;   // The model of the network.

        this.inputLayer = createLayer(inputSize);
        this.outputLayer = createLayer(outputSize, true);
        this.hiddenLayers = new Array(hiddenLayersNum);
        this.learningRate = option.learningRate || 0.5;

        // Create hidden layers.
        if (Array.isArray(hiddenLayersSize)) {
            for (let i = 0; i < model.hiddenLayers.length; i++) {
                model.hiddenLayers[i] = createLayer(hiddenLayersSize[i]);
            }
        } else if (typeof hiddenLayersSize === 'number') {
            for (let i = 0; i < model.hiddenLayers.length; i++) {
                model.hiddenLayers[i] = createLayer(hiddenLayersSize);
            }
        } else {
            throw new Error("option.hiddenLayers.size must be an array or number.");
        }

        // Set up weights between input layer & the first hidden layer randomly.
        createRandomWeights(model.inputLayer, model.hiddenLayers[0].length);

        // Set up weights among hidden layers
        // & between the last hidden layer & output layer
        model.hiddenLayers.forEach((layer, i) => {
            var nextLayerNum = (i != model.hiddenLayers.length - 1) ?   // Not the last hidden layer?
                model.hiddenLayers[i + 1].length :
                model.outputLayer.length;

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
    function createLayer(layerSize, output = false) {
        const layer = Array.apply(null, Array(layerSize)).map(() => {
            return output !== true ? 
                { weights: [], value: 0 } :
                { value: 0 }    // Output layer has no weight.
        });

        layer.bias = output !== true ? 1 : .35;
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
        for (let neuron of layer) {
            for (let i = 0; i < nextLayerNum; i++) {
                neuron.weights[i] = random.normal(0, 1);
                //neuron.weights[i] = Math.random() * 2 - 1;
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

    /**
     * Update every neuron's output value in a layer.
     * Used in forward propagation.
     * 
     * @private
     * @param {Array} layer the layer to be updated. 
     * @param {Array} preLayer the previous layer.
     */
    function updateLayer(layer, preLayer) {
        let index = 0;
        for (const neuron of layer) {
            neuron.value = 0;
            // Calculate sum(w * v)
            for (const preNeuron of preLayer) {
                neuron.value += preNeuron.value * preNeuron.weights[index];
            }
            // Apply activation function.
            let output = sigmoid(neuron.value + layer.bias);
            neuron.value = output;
            index++;
        }
    }

    /**
     * Update each neuron's error, which is the total error's
     * partial derivative of this neuron's input.
     * Used in backward propagation.
     * @private
     * @param {Array} outputLayer the output layer of the network. 
     * @param {Array} actual the actual values the `layer` should contain.
     */
    function updateOutputLayerError(outputLayer, actual) {
        let i = 0;
        for (const neuron of outputLayer) {
            // Total error's partial derivative of each output neuron's input.
            neuron.error = -(actual[i] - neuron.value) * neuron.value * (1 - neuron.value);
            i ++;
        }
    }

    /**
     * Update each neuron's error in `layer`.
     * Called in `train` method, after `updateOutputLayerError`.
     * Note that this `layer` cannot be the last layer of
     * the network.
     * @private
     * @param {Array} layer a layer to be updated.
     * @param {Array} nextLayer the previously updated layer.
     */
    function updateOneLayerError(layer, nextLayer) {
        for (const neuron of layer) {
            neuron.errors = [];     // weights error. Used for adjusting weights.

            // use to calculate the total error's partial
            // derivative of the current neuron's input.
            let error = 0, i = 0;
            for (const nextNeuron of nextLayer) {
                neuron.errors.push(nextNeuron.error * neuron.value);
                error += nextNeuron.error * neuron.weights[i];
                i++;
            }
            // The total error's partial derivative of this neuron's input.
            neuron.error = error * neuron.value * (1 - neuron.value);
        }
    }

    /**
     * Update every neuron's weights in `layer`.
     * Called after updateLayerError.
     * @private
     * @param {Array} layer 
     * @param {Number} learningRate 
     */
    function updateWeights(layer, learningRate) {
        for (const neuron of layer) {
            let i = 0;
            for (const error of neuron.errors) {
                neuron.weights[i] -= error * learningRate;
                i++;
            }
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
        if (input.length !== this.inputLayer.length) {
            throw new Error('Length of input\'s array dose not match.');
        }
        var index = 0, result = [];

        // Update every neuron's value in input layer.
        for (const neuron of this.inputLayer) {
            neuron.value = input[index++];
        }
        

        // Update the hidden layers.
        updateLayer(this.hiddenLayers[0], this.inputLayer);
        for (let i = 1; i < this.hiddenLayers.length; i++) {
            updateLayer(this.hiddenLayers[i], this.hiddenLayers[i - 1]);
        }

        // Update the output layer.
        updateLayer(this.outputLayer, this.hiddenLayers[this.hiddenLayers.length - 1]);
        
        // Obtain result.
        for (const neuron of this.outputLayer) {
            result.push(neuron.value);
        }

        return result;
    }

    /**
     * Use this function to train the network.
     * 
     * @param {Array} input Input to the network. 
     * @param {Array} output Actual output.
     */
    NeuralNet.prototype.learnSingle = function(input, output) {
        const hiddenLayersNum = this.hiddenLayers.length;
        const hiddenLayers = this.hiddenLayers;
        const outputLayer = this.outputLayer;
        const inputLayer = this.inputLayer;

        // Forward pass.
        this.totalError = totalError(this.predict(input), output);

        // Backward pass.
        updateOutputLayerError(outputLayer, output);
        updateOneLayerError(hiddenLayers[hiddenLayersNum - 1], outputLayer);
        for (let i = hiddenLayersNum - 2; i >= 0; i--) {
            updateOneLayerError(hiddenLayers[i], hiddenLayers[i + 1]);
        }
        updateOneLayerError(inputLayer, hiddenLayers[0]);

        // Update weights.
        updateWeights(inputLayer, this.learningRate);
        for (const layer of hiddenLayers) {
            updateWeights(layer, this.learningRate);
        }
    }

    /**
     * 
     */
    NeuralNet.prototype.learn = function (train_data) {
        // Learn it a couple of times.
        for (let i = 0; i < 10000; i++)
            for (const example of train_data) {
                this.learnSingle(example.input, example.output);
            }
    }
    return NeuralNet;
})();