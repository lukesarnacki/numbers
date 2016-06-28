require 'rubystats/normal_distribution'
require 'byebug'
require 'csv'

class Network
  attr_accessor :layers

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
    Tools.max_index(calculate_output(input))
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
      #save_weights_and_bias
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

  def copy
    new_network = self.dup
    new_network.layers = layers.map do |layer|
      layer.map { |neuron| neuron.dup }
    end

    new_network
  end
end

class NetworkTrainer
  attr_reader :network,
    # Wspolczynnik uczenia
    :learning_rate,
    # Maksymalna liczba epok
    :max_epochs,
    # Ilosc wierszy treningowych wykorzystywanych w kazdej epoce
    :batch_size,
    # W trybie multi nie są wypisywane poszczegolne epoki, tylko najlepszy wynik
    # (przydatne przy porownywaniu jakosci modelu dla roznych parametrow sieci i
    # uczenia
    :multi

    attr_accessor :max_training_progress, :current_training_progress, :best_network

  def initialize(network, learning_rate: 3.0, max_epochs: 30, batch_size: 20, multi: false)
    @network = network
    @learning_rate = learning_rate
    @max_epochs = max_epochs
    @batch_size = batch_size
    @multi = multi
    @max_training_progress = 0
  end

  def random_batch
    training_data.shuffle[0..batch_size]
  end

  def training_data
    @training_data ||= DataLoader.new(test: false).load
  end

  def training_data_size
    training_data.length
  end

  # Zwraca tablice w postaci [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] - jest to oczekiwane
  # wyjście sieci dla cyfry przekazanej jako parametr.
  def expected_output(digit)
    Array.new(network.output_layer_size, 0.0).insert(digit, 1.0)
  end

  def training_progress(network = nil)
    network ||= self.network
    training_data.select{|row| row[:output] == network.digit(row[:input]) }.count / training_data_size.to_f
  end

  def set_best_network
    self.current_training_progress = training_progress

    if current_training_progress > max_training_progress
      self.max_training_progress = current_training_progress
      self.best_network = network.copy
    end
  end

  def train
    max_epochs.times.each do |epoch|
      random_batch.each_with_index do |row, index|
        network.calculate_output(row[:input])
        network.train(expected_output(row[:output]), learning_rate)

        print "Row: #{index}/#{batch_size}\r" unless multi
      end

      network.save_weights_and_bias
      set_best_network

      print "Epoch: #{epoch}/#{max_epochs}\r" if multi
      puts "Epoch: #{epoch}: #{(current_training_progress * 100.0).round(2)}%" unless multi
    end
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
    @new_bias ||= bias
    @new_bias = bias + delta_b
  end

  def save_weights_and_bias
    @weights = @new_weights
    @bias = @new_bias
    @new_weights = nil
    @new_bias = nil
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

  def self.max_index(array)
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

network = Network.new([64, 100, 10])
trainer = NetworkTrainer.new(network, learning_rate: 1.8, max_epochs: 50, batch_size: 200)
trainer.train
trained_network = trainer.best_network

data = DataLoader.new(test: true).load

all = data.count
good = 0
data.each do |row|
  good += 1 if trained_network.digit(row[:input]) == row[:output]
end

puts "Result: #{good}/#{all}"

# neurons = [10, 40, 50, 100]
# learning_rates = [1.5, 2.5]
# batch_sizes = [20, 50, 100, 200]
#
# data = DataLoader.new(test: true).load
#
# neurons.each do |n_count|
#   learning_rates.each do |learning_rate|
#     batch_sizes.each do |batch_size|
#       network = Network.new([64,n_count,10])
#       trainer = NetworkTrainer.new(network, learning_rate: learning_rate, max_epochs: 50, batch_size: batch_size, multi: true)
#       trainer.train
#       network = trainer.best_network
#       good = data.select { |row| network.digit(row[:input]) == row[:output] }.count
#       p  [:neurons_count, n_count, :batch_size, batch_size, :learning_rate, learning_rate, :good, good]
#     end
#   end
# end
