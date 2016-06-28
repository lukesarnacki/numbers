require 'rubystats/normal_distribution'
require 'byebug'
require 'csv'

class Network
  attr_reader :layers

  def initialize(neurons_in_layers)
    @layers = []
    neurons_in_layers[1..-1].each_with_index do |neurons_number, layer|
      inputs_number = neurons_in_layers[layer]
      @layers[layer] = neurons_number.times.map do
        Neuron.new(inputs_number: inputs_number)
      end
    end
  end

  def digit(input)
    Tools.max(calculate_output(input))
  end

  def output_layer_size
    layers[-1].count
  end

  def train(expected_output, learning_rate, layer = -1)
    layers[layer].each_with_index.map do |neuron, neuron_number|
      inputs = neuron.inputs

      inputs.each_with_index do |input, i|
        error = if layer == -1
                  # Dla warstwy wyjsciowej blad jest liczony na podstawie
                  # oczekiwanej odpowiedzi
                  Tools.sigmoid_prime(neuron.net) * (expected_output[neuron_number] - neuron.output)
                else
                  # Dla warstw ukrytych blad jest liczony na podstawie bledow
                  # nastepnej warstwy
                  next_layer = layer + 1
                  # Tablica wag które wychodzą z aktualnie rozpatrywanego
                  # neuronu do neuronow z kolejnej warstwy
                  weights = layers[next_layer].map{|n| n.weights[neuron_number]}
                  # Bledy neuronow nastepnej warstwy
                  errors = layers[next_layer].map(&:error)
                  Tools.sigmoid_prime(neuron.net) * Tools.multiply_arrays(weights, errors)
                end

        neuron.error = error
        delta_w = learning_rate * error * input
        delta_b = learning_rate * error
        neuron.update_weight(i, delta_w)
        neuron.update_bias(delta_b)
      end
    end

    if layers.count + layer == 0
      # To byla pierwsza warstwa ukryta, dlatego mozemy zapisac wagi i bias
      save_weights_and_bias
      return
    end
    train(expected_output, learning_rate, layer - 1)
  end

  def save_weights_and_bias
    layers.each {|layer| layer.map { |neuron| neuron.save_weights_and_bias } }
  end

  def calculate_output(inputs, layer: -1)
    output = if(layer == -1)
      inputs
    else
      layers[layer].map {|neuron| neuron.calculate_output(inputs) }
    end

    return output if layer >= layers.count - 1

    calculate_output(output, layer: layer + 1)
  end
end

class NetworkTrainer
  attr_reader :network, :learning_rate, :max_epochs, :batch_size, :multi

  def initialize(network, learning_rate: 3.0, max_epochs: 30, batch_size: 20, multi: false)
    @network = network
    @learning_rate = learning_rate
    @max_epochs = max_epochs
    @batch_size = batch_size
    @multi = multi
  end

  def train
    data = DataLoader.new(test: false).load
    good = []
    max_epochs.times.each do |epoch|
      data.shuffle[0..batch_size].each_with_index do |row, index|
        input = row[:input]
        expected_output = Array.new(network.output_layer_size, 0.0).insert(row[:output], 1.0)
        network.calculate_output(input)
        network.train(expected_output, learning_rate)
        print "Row: #{index}/#{batch_size}\r" unless multi
      end

      all = data.count
      good << data.select{|row| row[:output] == network.digit(row[:input]) }.count

      print "Epoch: #{epoch}/#{max_epochs}\r" if multi
      puts "Epoch: #{epoch}: #{good.last}/#{all}" unless multi
    end

    "epoch #{good.each_with_index.max[1]}, good #{good.max}"
  end
end


class Neuron
  attr_reader :inputs_number, :weights, :bias, :output, :inputs
  attr_accessor :error

  def initialize(inputs_number: )
    @inputs_number = inputs_number
    @weights = initialize_weights
    @bias = Tools.random
    @error = 0.0
  end

  def calculate_output(inputs)
    @inputs = inputs
    @output = Tools.sigmoid(net + bias)
  end

  def net
    Tools.multiply_arrays(weights, inputs)
  end

  def update_weight(i, delta_w)
    @new_weights ||= weights
    @new_weights[i] = weights[i] + delta_w
  end

  def update_bias(delta_b)
    @new_bias = bias + delta_b
  end

  def save_weights_and_bias
    @weights = @new_weights
    @bias = @new_bias
  end

  private

  def initialize_weights
    inputs_number.times.map { Tools.random }
  end
end

class NeronsLoader
  def initialize(neurons_count: , inputs_number:)
  end
end

class Tools
  # Losowa liczba rozkładu normalnego
  def self.random
    Rubystats::NormalDistribution.new(0,1).rng
  end

  def self.sigmoid(z)
    1.0 / (1.0 + Math.exp(-z))
  end

  def self.sigmoid_prime(z)
    sigmoid(z) * (1.0 - sigmoid(z))
  end

  def self.multiply_arrays(a, b)
    if a.count != b.count
      raise "arrays are not the same size:\na is #{a.count} and b is #{b.count}\na: #{a.inspect}\nb: #{b.inspect}"
    end

    a.length.times.map { |i| a[i] * b[i] }.inject(0){|sum,x| sum + x }
  end

  def self.max(array)
    array.each_with_index.max[1]
  end
end

class DataLoader

  attr_reader :test

  def initialize(test: false)
    @test = test
  end

  def load
    result = []
    CSV.foreach("data/optdigits.#{test ? 'tes' : 'tra'}") do |row|
      hash = {}
      hash[:input] = row[0..63].map{ |e| e.to_f }.flatten
      hash[:output] = row[64].to_i
      result << hash
    end

    result
  end
end

data = DataLoader.new(test: true).load

network = Network.new([64, 31,21,10])
good = NetworkTrainer.new(network, learning_rate: 3.0, max_epochs: 30, batch_size: 200).train

data = DataLoader.new.load
all = data.count
good = 0
data.each do |row|
  good += 1 if network.digit(row[:input]) == row[:output]
end

puts "Result: #{good}/#{all}"


# neurons = 7..16
# learning_rates = [1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.0]
# batch_sizes = [20, 30, 40, 50, 80]
#
# neurons.each do |n_count|
#   learning_rates.each do |learning_rate|
#     batch_sizes.each do |batch_size|
#       network = Network.new([64,n_count,10])
#       good = NetworkTrainer.new(network, learning_rate: learning_rate, max_epochs: 50, batch_size: batch_size, multi: true).train
#       p  [:neurons_count, n_count, :batch_size, batch_size, :learning_rate, learning_rate, :good, good]
#     end
#   end
# end
