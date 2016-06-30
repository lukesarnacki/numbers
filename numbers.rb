require 'rubystats/normal_distribution'
require 'byebug'
require 'csv'
require 'celluloid/current'
require 'bigdecimal'

DESIRED_QUALITY = 0.9
MAX_EPOCHS = 500

class Network
  attr_accessor :layers

  def initialize(neurons_in_layers)
    @layers = []
    neurons_in_layers[1..-1].each_with_index do |neurons_number, layer|
      inputs_number = neurons_in_layers[layer]
      # Wypelnianie warst neuronami. Tylko warstwy ukryte i warstwa wyjsciowa sa
      # reprezentowane jako neurony, warstwa wejsciowa nie ma reprezentacji w
      # kodzie
      @layers[layer] = neurons_number.times.map do
        Neuron.new(inputs_number: inputs_number)
      end
    end
  end

  # Dla danych wejsciowych podanych jako parametr, metoda zwraca jako jaka cyfre
  # te dane zostaly sklasyfikowane.
  def digit(input)
    Tools.max_index(calculate_output(input))
  end

  def output_layer_size
    layers[-1].count
  end

  # Rekurencyjnie trenuje siec dla wszystkich warst zaczynajac od ostatniej
  # (indeks -1) a konczac na pierwszej warstwie ukrytej.
  def train(inputs, expected_output, learning_rate, layer = -1)
    if layer == -1
      calculate_output(inputs) # przed obliczeniami ustawia wejscia wszystkich neuronow sieci dla danych wejsciowych
    end

    layers[layer].each_with_index.map do |neuron, neuron_number|
      neuron.error = if layer == -1
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

      # q * d
      delta = learning_rate * neuron.error

      neuron.update_bias(delta)

      neuron.inputs.each_with_index do |input, i|
        # q * d * z
        neuron.update_weight(i, delta * input)
      end
    end

    if layers.count + layer == 0
      # To byla ostatnia warstwa, mozna zakonczyc wykonanie metody
      return
    end
    train(inputs, expected_output, learning_rate, layer - 1)
  end

  def save_weights_and_bias(learning_rate, batch_size)
    layers.each {|layer| layer.map { |neuron| neuron.save_weights_and_bias(learning_rate, batch_size) } }
  end

  # Oblicza wyjscie sieci dla danych wejsciowych, propagujac wejscie na
  # wszystkie warstwy. Zwraca tablice o dlugosci rownej ilosci neuronow w
  # ostatniej warstwie.
  def calculate_output(inputs, layer: 0)
    output = layers[layer].map {|neuron| neuron.calculate_output(inputs) }

    # Dla warstwy wyjsciowej zwracamy obliczone wartosci wyjsciowe
    return output if layer >= layers.count - 1

    # Wywoalenie rekurencyjne metody dla kolejnej warstwy
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

    attr_accessor :max_properly_classified, :current_properly_classified, :best_network

  def initialize(network, learning_rate: 3.0, max_epochs: 30, batch_size: 10, multi: false)
    @network = network
    @learning_rate = learning_rate
    @max_epochs = max_epochs
    @batch_size = batch_size
    @multi = multi
    @max_properly_classified = 0
  end

  # Wybiera sposrod danych zestaw danych dla danej epoki. Jesli minie tyle epok,
  # ze dane sie skoncza, to uklada dane w tablicy losowo i zaczyna jeszcze raz
  def batch(epoch)
    shuffle_training_data if epoch * batch_size > training_data.size

    from = (epoch * batch_size) % training_data.size
    to = [from + batch_size - 1, training_data.size - 1].min
    training_data[from..to]
  end

  # Uklada dane treningowe w losowej kolejnosci
  def shuffle_training_data
    training_data.shuffle!
  end

  def data
    @data ||= DataLoader.new(test: false).load.shuffle
  end

  # Dane treningowe ustawione w tablicy losowo
  def training_data
    @training_data ||= data[1000..-1]
  end

  # Dane do walidacji ustawione w tablicy losowo
  def validation_data
    @validation_data ||= data[0..999]
  end

  # Zwraca tablice w postaci [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] - jest to oczekiwane
  # wyjście sieci dla cyfry przekazanej jako parametr.
  def expected_output(digit)
    Array.new(network.output_layer_size, 0.0).insert(digit, 1.0)
  end

  def properly_classified(network = nil)
    network ||= self.network
    BigDecimal.new(validation_data.select{|row| row[:output] == network.digit(row[:input]) }.count)
  end

  # Stosunek wartosci sklasyfikowanych poprwanie do wszystkich wartosci w
  # zbiorze danych treningowych
  def training_quality
    BigDecimal.new(current_properly_classified) / BigDecimal.new(validation_data.count)
  end

  def network_error
    BigDecimal.new(1) - self.training_quality
  end

  # Jesli aktualna jakosc jest najlepsza, siec neuronowa z aktualnymi
  # parametrami neuronow jest zapisywana jako "najlepsza" (nie zawsze
  # siec wytrenowana w ostatniej epoce bedzie miala najwieksza jakosc)
  def set_best_network
    self.current_properly_classified = properly_classified

    if current_properly_classified > max_properly_classified
      self.max_properly_classified = current_properly_classified
      self.best_network = network.copy
    end
  end

  # Dla kazdej epoki wybieramy losowe n wartosci (ilosc zdefiniowana przez
  # batch_size), ktore beda sluzyly do trenowania sieci w danej epoce
  def train
    puts "Training started: neurons_count: #{network.layers[0].count}, learning_rate: #{learning_rate}, batch_size: #{batch_size}"
    max_epochs.times.each do |epoch|
      batch(epoch).each_with_index do |row, index|
        network.train(row[:input], expected_output(row[:output]), learning_rate)

        print "Row: #{index+1}/#{batch_size}\r" unless multi
      end

      # Po zakonczeniu epoki, zapisywane sa wagi i bias oraz wywolywana jest
      # metoda, ktora ustawia aktualna siec jako ta najlepsza, jesli jej jakosc
      # jest najwieksza z dotychczasowych epok
      network.save_weights_and_bias(learning_rate, batch_size)
      set_best_network

      results << { error: network_error.to_f, properly_classified: current_properly_classified.to_i }

      #print "Epoch: #{epoch}/#{max_epochs}\r" if multi
      puts "Epoch: #{epoch}: #{(training_quality * BigDecimal.new(100)).to_f.round(3)}, #{network_error.to_f.round(4)}" unless multi

      # Jesli model osiagnie oczekiwana jakosc, mozna zakonczyc trening
      # return results if training_quality >= DESIRED_QUALITY
    end
    puts "Training ended: neurons_count: #{network.layers[0].count}, learning_rate: #{learning_rate}, batch_size: #{batch_size}"

    return results
  end

  def results
    @results ||= []
  end
end


class Neuron
  attr_reader :inputs_number, :weights, :bias, :output, :inputs
  attr_accessor :error

  def initialize(inputs_number: )
    @inputs_number = inputs_number
    @weights = initialize_weights
    @bias = Tools.random
    @error = nil
  end

  # Wyjscie neuronu
  def calculate_output(inputs)
    @inputs = inputs
    @output = Tools.sigmoid(net + bias)
  end

  # Iloczyn wag i wartosci na wejsciach + bias
  def net
    Tools.multiply_arrays(weights, inputs)
  end

  # Uaktualnia wage, ale jeszcze jej nie zapisuje
  def update_weight(i, delta)
    @w_deltas ||= []
    @w_deltas[i] ||= []
    @w_deltas[i] << delta
  end

  # Uaktualnia bias, ale jeszcze go nie zapisuje
  def update_bias(delta)
    @b_deltas ||= []
    @b_deltas << delta
  end

  # Zapisuje uaktualnione wagi i bias jako aktualne
  def save_weights_and_bias(learning_rate, batch_size)
    @weights = @weights.each_with_index.map do |w, i|
      w + (@w_deltas[i].reduce(&:+).to_f / @w_deltas[i].size)
      #w - ((learning_rate.to_f/batch_size.to_f) * @w_deltas[i].reduce(:+))
    end
    @bias = @bias + (@b_deltas.reduce(&:+).to_f / @b_deltas.size)
    #@bias = @bias - ((learning_rate.to_f/batch_size.to_f) * @b_deltas.reduce(:+))
    @w_deltas = []
    @b_deltas = []
  end

  private

  # Inicjalizacja wag za pomoca zmienny generowanych w rozkladzie normalnym
  def initialize_weights
    inputs_number.times.map { Tools.random }
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

class Tester
  attr_reader :neurons_count, :learning_rate, :batch_size, :multi

  def initialize(neurons_count, learning_rate, batch_size, multi: false)
    @neurons_count = neurons_count
    @learning_rate = learning_rate
    @batch_size = batch_size
    @multi = multi
  end

  def network
    @network ||= Network.new([64,neurons_count,10])
  end

  def trainer
    @trainer ||= NetworkTrainer.new(network, learning_rate: learning_rate, max_epochs: MAX_EPOCHS, batch_size: batch_size, multi: multi)
  end

  def test_data
    @test_data ||= DataLoader.new(test: true).load
  end

  def properly_classified(network)
    @properly_classified ||= BigDecimal.new(test_data.select { |row| network.digit(row[:input]) == row[:output] }.count)
  end

  def error(network)
    1 - (properly_classified(network)/BigDecimal.new(test_data.count))
  end

  def csv_headers(network)
    ["ilosc danych testowych", test_data.count, "blad sieci", error(network).to_f, "przypadki sklasyfikowane dobrze", properly_classified(network).to_i]
  end

  def test_network
    results = trainer.train
    network = trainer.best_network

    CSV.open("csv/#{neurons_count.to_i}_#{learning_rate.to_i}_#{batch_size.to_i}.csv", "wb") do |csv|
      csv << csv_headers(network)
      results.each {|result| csv << [result[:error], result[:properly_classified]] }
     end

    p  [:neurons_count, neurons_count, :batch_size, batch_size, :learning_rate, learning_rate, error(network).to_f]
  end
end

class TesterWorker
  include Celluloid

  def test_network(neurons_count, learning_rate, batch_size)
    Tester.new(neurons_count, learning_rate, batch_size, multi: true).test_network
  end
end

#Tester.new(50, 5.0, 50).test_network

neurons_counts = [10, 50, 100]
learning_rates = [3.0, 5.0 , 7.0]
batch_sizes = [50, 100, 200]

pool = TesterWorker.pool(size: 50)

workers = []

# Przypadki raczej slabo uczace
workers.push pool.future.test_network(10, 3.0, 10)
workers.push pool.future.test_network(50, 3.0, 10)

neurons_counts.map do |neurons_count|
  learning_rates.map do |learning_rate|
    batch_sizes.map do |batch_size|
      workers.push pool.future.test_network(neurons_count, learning_rate, batch_size)
    end
  end
end

workers.each { |worker| worker.value }

# network = Network.new([64,40,10])
# trainer = NetworkTrainer.new(network, learning_rate: 5.0, max_epochs: 500, batch_size: 10)
# trainer.train
# trained_network = trainer.best_network
#
# data = DataLoader.new(test: true).load
#
# all = data.count
# good = 0
# data.each do |row|
#   good += 1 if trained_network.digit(row[:input]) == row[:output]
# end
#
# puts "Result: #{good}/#{all}"
