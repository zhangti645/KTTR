import numpy as np
import math
from collections import Counter
import itertools


### 0-order
def get_digits(num):
    result = list(map(int, str(num)))
    return result


def map_signed_to_unsigned(n):
    # Map signed integer to unsigned integer
    if n >= 0:
        return 2 * n
    else:
        return -2 * n - 1


def map_unsigned_to_signed(n):
    # Map unsigned integer back to signed integer
    if n % 2 == 0:
        return n // 2
    else:
        return -(n // 2) - 1


def exponential_golomb_encode(n):
    # Map signed integer to unsigned integer before encoding
    n = map_signed_to_unsigned(n)

    unarycode = ''
    golombCode = ''

    # Quotient and Remainder Calculation
    groupID = np.floor(np.log2(n + 1))
    temp_ = groupID

    while temp_ > 0:
        unarycode += '0'
        temp_ -= 1

    index_binary = bin(n + 1).replace('0b', '')
    golombCode = unarycode + index_binary
    return golombCode


def exponential_golomb_decode(golombcode):
    code_len = len(golombcode)

    # Count the number of '1's followed by the first '0'
    m = 0
    for i in range(code_len):
        if golombcode[i] == '0':
            m += 1
        else:
            ptr = i  # First '1' position after '0's
            break

    offset = 0
    for ii in range(ptr, code_len):
        num = int(golombcode[ii])
        offset += num * (2 ** (code_len - ii - 1))

    decodemum = offset - 1

    # Map back to signed integer after decoding
    return map_unsigned_to_signed(decodemum)


def expgolomb_split(expgolomb_bin_number):
    x_list = expgolomb_bin_number
    del (x_list[0])  # Remove the starting identifier to avoid issues with 0
    x_len = len(x_list)

    sublist = []
    while len(x_list) > 0:
        count_number = 0
        i = 0
        if x_list[i] == 1:
            sublist.append(x_list[0:1])
            del (x_list[0])
        else:
            num_times_zeros = [len(list(v)) for k, v in itertools.groupby(x_list)]
            count_number += num_times_zeros[0]
            sublist.append(x_list[0:(count_number * 2 + 1)])
            del (x_list[0:(count_number * 2 + 1)])
    return sublist


# Main Execution
if __name__ == "__main__":
    # Example binary sequence
    x = 10100111100010110010000111101100101011

    x_list = get_digits(x)
    del (x_list[0])  # Remove starting identifier

    sublist = []
    while len(x_list) > 0:
        count_number = 0
        i = 0
        if x_list[i] == 1:
            sublist.append(x_list[0:1])
            del (x_list[0])
        else:
            num_times_zeros = [len(list(v)) for k, v in itertools.groupby(x_list)]
            count_number += num_times_zeros[0]
            sublist.append(x_list[0:(count_number * 2 + 1)])
            del (x_list[0:(count_number * 2 + 1)])

    print("Encoded segments:", sublist)
    print("Number of segments:", len(sublist))

    # Decode each segment and print the result
    for i in range(len(sublist)):
        decoded_num = exponential_golomb_decode(sublist[i])
        print("Decoded number:", decoded_num)
