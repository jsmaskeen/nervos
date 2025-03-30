from nervos.examples.neuron_spiking import ExampleCurrentThroughLIFNeuron
from nervos.utils import Parameters

p = Parameters()
p.from_url('https://pastebin.com/raw/7F4wUc0D')

sim = ExampleCurrentThroughLIFNeuron(p)
sim.simulate_pulse(180)
