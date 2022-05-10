import fileinput
import math
import random
import matplotlib.pyplot as plt
from statistics import mean
from statistics import variance
import numpy as np


class Demon:
    model = []
    model_pad = []

    def __init__(self, size, energy=10):
        if isinstance(size, int):
            self.model_size = (size, size)
        elif isinstance(size[0], int) and isinstance(size[1], int):
            self.model_size = size
        else:
            self.model_size = (10, 10)
        self.energy_fill = energy
        self.model_energy = 0
        self.non_abs_magnetization = 0
        self.magnetization = 0

    def create_model(self, porb_of_one):
        if porb_of_one > 1:
            porb_of_one = 1

        self.model = np.zeros((self.model_size[0], self.model_size[1]))
        prob_of_one_str = str(porb_of_one)

        range_up = 10
        for char in prob_of_one_str[2:len(prob_of_one_str)]:
            if char == '0':
                range_up *= 10

        for i in range(self.model_size[0]):
            for j in range(self.model_size[1]):
                if porb_of_one == 0:
                    self.model[i, j] = -1
                else:
                    random_int = random.randint(0, range_up)
                    if random_int <= porb_of_one * range_up:
                        self.model[i, j] = 1
                    else:
                        self.model[i, j] = -1
        self.model_energy = self.comp_energy()
        self.magnetization = self.comp_magnetization()
        self.copy_model_with_padding()

    def copy_model_with_padding(self):
        i_size = self.model_size[0]
        j_size = self.model_size[1]
        self.model_pad = np.zeros((i_size + 2, j_size + 2))
        padding_rows = np.zeros((2, j_size + 2))
        padding_cols = np.zeros((2, i_size + 2))

        padding_rows[0, 1:i_size + 1] = self.model[i_size - 1, :]
        padding_rows[1, 1:i_size + 1] = self.model[0, :]
        padding_cols[0, 1:j_size + 1] = self.model[:, j_size - 1]
        padding_cols[1, 1:j_size + 1] = self.model[:, 0]

        self.model_pad[0] = padding_rows[0]
        self.model_pad[i_size + 1] = padding_rows[1]
        self.model_pad[:, 0] = padding_cols[0]
        self.model_pad[:, j_size + 1] = padding_cols[1]
        for i in range(i_size):
            self.model_pad[i + 1, 1:i_size + 1] = self.model[i, :]

    # row and col - indices of padding model changed state
    def update_padding_after_state_change(self, row, col):
        if row == 1:
            self.model_pad[self.model_size[0] + 1, col] = self.model_pad[row, col]
        elif row == self.model_size[0]:
            self.model_pad[0, col] = self.model_pad[row, col]
        if col == 1:
            self.model_pad[row, self.model_size[1] + 1] = self.model_pad[row, col]
        elif col == self.model_size[1]:
            self.model_pad[row, 0] = self.model_pad[row, col]

    def comp_energy(self):
        energy = 0
        for i in range(0, self.model_size[0]):
            for j in range(0, self.model_size[1]):
                if j != self.model_size[1] - 1:
                    energy += self.model[i, j] * self.model[i, j + 1]
                else:
                    energy += self.model[i, 0] * self.model[i, j]
                if i != self.model_size[0] - 1:
                    energy += self.model[i, j] * self.model[i + 1, j]
                else:
                    energy += self.model[0, j] * self.model[i, j]
        return energy * -1

    def comp_energy_for_one_state(self, row, col):
        energy = 0
        energy += self.model_pad[row, col] * self.model_pad[row, col - 1]
        energy += self.model_pad[row, col] * self.model_pad[row + 1, col]
        energy += self.model_pad[row, col] * self.model_pad[row, col + 1]
        energy += self.model_pad[row, col] * self.model_pad[row - 1, col]
        return energy

    def comp_max_energy(self):
        temp_demon = Demon(self.model_size)
        temp_demon.model = np.ones((self.model_size[0], self.model_size[1]))
        return abs(temp_demon.comp_energy())

    def execute_state_change(self, row, col, state, difference):
        self.model_energy *= -1
        self.model_energy += difference
        self.energy_fill += difference
        self.model_energy *= -1
        self.model[row, col] = state
        self.non_abs_magnetization += state * 2
        self.magnetization = abs(self.non_abs_magnetization)

    def change_model_state(self, row, col, state):
        if state == 1 or state == -1:
            current_state = self.model[row, col]
            current_cell_energy = self.comp_energy_for_one_state(row + 1, col + 1)
            self.model_pad[row + 1, col + 1] = state
            self.update_padding_after_state_change(row + 1, col + 1)
            new_cell_energy = self.comp_energy_for_one_state(row + 1, col + 1)

            difference = new_cell_energy - current_cell_energy
            if difference > 0:
                self.execute_state_change(row, col, state, difference)
                return
            else:
                if abs(difference) <= self.energy_fill:
                    self.execute_state_change(row, col, state, difference)
                else:
                    self.model_pad[row + 1, col + 1] = current_state
                    self.update_padding_after_state_change(row + 1, col + 1)

        else:
            print("Invalid state value")
            return

    def comp_magnetization(self):
        magnetization = 0
        for i in range(self.model_size[0]):
            for state in self.model[i]:
                magnetization += state
        self.non_abs_magnetization = magnetization
        return abs(magnetization)


def comp_lin_reg_a(data_distrib, values):
    data_sum_xy = 0
    data_sum_y = 0
    for val in values:
        data_sum_xy += math.log(data_distrib[val]) * val
        data_sum_y += math.log(data_distrib[val])
    mean_xy = data_sum_xy / len(values)
    mean_x = mean(values)
    mean_y = data_sum_y / len(values)

    return (mean_xy - mean_x * mean_y) / variance(values)


def comp_lin_reg_b(data_distrib, values, a_factor):
    print("a fac", a_factor)
    data_sum_x = 0
    data_sum_y = 0
    for val in values:
        data_sum_x += val
        data_sum_y += math.log(data_distrib[val])
    mean_x = data_sum_x / len(values)
    mean_y = data_sum_y / len(values)
    return mean_y - (a_factor * mean_x)


def comp_temp(a_factor):
    return -1 / a_factor


input_line = [line for line in fileinput.input()]
input_data = []
num = ''
for i in input_line[0]:
    if str.isspace(i):
        input_data.append(int(num))
        num = ''
        continue
    else:
        num += i

print("Input: ", input_data)

row_size = int(input_data[0])
col_size = int(input_data[1])
time_steps = int(input_data[2])
runs = int(input_data[3])
start_energies = [int(energy) for energy in input_data[4:]]

magnets_for_run = []
temperatures = []
print(runs)
for run in range(int(runs)):
    print("Run", run)
    print("Start demon energy", start_energies[run])
    lil_demon = Demon((row_size, col_size), start_energies[run])
    lil_demon.create_model(1)
    demon_energy = []
    print("Start model energy: ", lil_demon.model_energy)
    print(f"Start magnetisation: {lil_demon.magnetization}")
    en_val = []
    occurrence = np.zeros(lil_demon.energy_fill * 2)
    representative_data_idx = int(abs(lil_demon.energy_fill) / 2) + 10
    en_magnets_dict = dict()
    magnets = []
    idx = 0
    for idx in range(representative_data_idx + time_steps):
        if idx > representative_data_idx:
            magnets.append(lil_demon.magnetization)
            demon_energy.append(lil_demon.energy_fill)
            if int(lil_demon.energy_fill) in en_val:
                occurrence[int(lil_demon.energy_fill)] += 1
            else:
                en_val.append(int(lil_demon.energy_fill))
                en_val.sort()
                occurrence[int(lil_demon.energy_fill)] += 1

        row_idx = random.randint(0, lil_demon.model_size[0] - 1)
        col_idx = random.randint(0, lil_demon.model_size[1] - 1)
        lil_demon.change_model_state(row_idx, col_idx, lil_demon.model[row_idx, col_idx] * (-1))
        if lil_demon.magnetization == 0:
            break

    idx += 1

    magnets_mean = mean(magnets)
    if len(en_val) > 1:
        a_fact = comp_lin_reg_a(occurrence, en_val)
        temp = comp_temp(a_fact)
        print("Demon energy distribution:")
        for val in en_val:
            print(f"energy: {val} ; amount: {occurrence[val]}")
        temperatures.append(temp)
        print("Magnetisation mean: ", magnets_mean, "\n")
        magnets_for_run.append(magnets_mean)
    else:
        print(f"Non representative data, take new data (try to change parameters:\n"
              f"(Model size: {lil_demon.model_size})\nDemon energy: {start_energies[run]}"
              f"\nTime steps: {representative_data_idx})\n")

temp_demon = Demon((row_size, col_size), start_energies[0])
temp_demon.create_model(1)
temp_magnets = []
for idx in range(time_steps):
    temp_magnets.append(temp_demon.magnetization)
    row_idx = random.randint(0, temp_demon.model_size[0] - 1)
    col_idx = random.randint(0, temp_demon.model_size[1] - 1)
    temp_demon.change_model_state(row_idx, col_idx, temp_demon.model[row_idx, col_idx] * (-1))

plt.plot(np.arange(len(temp_magnets)), temp_magnets, 'o')
plt.xlabel("Time steps")
plt.ylabel("Magnetisation")
plt.show()

plt.plot(temperatures, magnets_for_run, 'o')
plt.xlabel("Temperature")
plt.ylabel("Magnetisation")
plt.show()

