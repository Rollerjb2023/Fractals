# Class for Julia Set image generation

# Image work
from PIL import Image, ImageDraw

# Multiprocessing
import concurrent.futures
from tqdm import tqdm

# Misc.
from typing import Callable
import numpy as np
from time import perf_counter
from fractal.utils import Fractal


class Julia(Fractal):
    def __init__(self, constant, coloring_scheme: Callable, iterations, real_width, imaginary_height,
                 image_dimensions: tuple[int, int], center: complex = 0+0j):
        """
        Constructs a Julia object
        :param constant: The constant applied during each iteration of (z**2 + c)
        :param coloring_scheme: A function that describes how to color the image based on the speed of divergence
        :type coloring_scheme: Callable[[int, int], tuple[int, int, int]]
        :param iterations: How many iterations of (z**2 + c) before assuming convergence
        :param real_width: The length of the real value interval
        :param imaginary_height: The length of the imaginary value interval
        :param image_dimensions: A tuple describing the pixel width and height of the image
        :param center: The complex point at the center of the image, defaults to 0+0j
        """

        self.constant = constant
        self.coloring_scheme: Callable[[int, int], tuple[int, int, int]] = coloring_scheme
        self.iterations = iterations
        self.real_width = real_width
        self.imaginary_height = imaginary_height
        self.center = center

        self.dimensions = image_dimensions
        self.width = self.dimensions[0]
        self.height = self.dimensions[1]

        # These serve optimization purposes
        # y of pixel where the real axis lies
        self.y_0 = self.height // 2 - int(self.center.imag * self.height / self.imaginary_height)
        # x of pixel where the imaginary axis lies
        self.x_0 = self.width // 2 - int(self.center.real * self.width / self.real_width)

    def __repr__(self):
        """
        Returns a string representation of the Julia object
        Is used as the default file name
        """
        return (f"Julia({self.constant}, {self.coloring_scheme.__name__}, {self.iterations}, {self.real_width}, "
                f"{self.imaginary_height}, {self.dimensions}, center={self.center})")

    def compute_pixel(self, point: complex):
        """
        Determines how many iterations a point requires before it is known to diverge (abs(p) > 2)
        Uses self.coloring_scheme to determine the right color for the point
        :param point: The complex point to be analyzed
        """

        value = point
        for z in range(0, self.iterations):
            value = value ** 2 + self.constant

            if abs(value) > 2:  # Diverges
                return self.coloring_scheme(z, self.iterations)

        return 0, 0, 0  # converges

    def generate(self, output_path="", draw_center_circle=False, file_name=""):
        """
        Computes and saves the Julia image, prepares and starts many processes
        Builds a 2D array of RGB tuples that gets used to generate an image
        :param output_path: The file path to which the finished image will be saved
        :param draw_center_circle: Set this to True to draw a small circle around the image center
        :param file_name: Use this to override the default naming behavior, see Julia.__repr__
        """

        # Multi-process, uses tqdm to generate a progress bar + ETA
        x_range = range(0, self.width)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = np.array(
                list(tqdm(executor.map(self.calculate_column, x_range), total=len(x_range))),
                dtype=np.uint8
            )

        # Single-process
        # results = np.array([self.draw(x) for x in range(self.width)], dtype=np.uint8)

        # Transposes the width-height sub-matrix
        transposed_results = np.transpose(results, (1, 0, 2))

        # Uses the rotational symmetry
        # copies symmetric part of the matrix, rotates it 180 degrees, then pastes it into the block previously skipped
        if 0 < self.x_0 < self.width and 0 < self.y_0 < self.height:  # Is the origin in the image?
            if self.y_0 <= self.height / 2:  # Is the origin in the upper or lower half of the image?
                if self.x_0 <= self.width / 2:  # Is the origin in the left or right half of the image?
                    block_to_copy = transposed_results[
                                    1: 2*self.y_0,
                                    1: self.x_0
                                    ]
                    transposed_results[
                        1: 2*self.y_0,
                        self.x_0+1: 2*self.x_0 + (self.center.real == 0)
                    ] = np.rot90(block_to_copy, 2)

                else:
                    block_to_copy = transposed_results[
                                    1-(self.center.imag == 0):2*self.y_0+1,
                                    2 * self.x_0 - self.width + 1: self.x_0
                                    ]
                    transposed_results[
                        0: 2*self.y_0,
                        self.x_0+1: self.width
                    ] = np.rot90(block_to_copy, 2)

            else:
                if self.x_0 <= self.width / 2:  # Is the origin in the left or right half of the image?
                    block_to_copy = transposed_results[
                                    2*self.y_0-self.height+1: self.height,
                                    1: self.x_0
                                    ]
                    transposed_results[
                        2*self.y_0-self.height+1: self.height,
                        self.x_0+1: 2*self.x_0
                    ] = np.rot90(block_to_copy, 2)

                else:
                    block_to_copy = transposed_results[
                                    2*self.y_0-self.height+1: self.height,
                                    2*self.x_0-self.width+1: self.x_0
                                    ]
                    transposed_results[
                        2*self.y_0-self.height+1: self.height,
                        self.x_0 + 1: self.width
                    ] = np.rot90(block_to_copy, 2)

        img = Image.fromarray(transposed_results, 'RGB')

        # Center circle, included for debugging purposes
        if draw_center_circle:
            ImageDraw.Draw(img).circle((self.width // 2, self.height // 2), radius=8)

        # Save image as png
        if output_path and output_path[-1] != "/":
            output_path = f"{output_path}/"
        if file_name:
            print(f"Saving image {file_name}.png to {output_path}.")
            img.save(f"{output_path}{file_name}.png")
        else:
            print(f"Saving image {self.__repr__()}.")
            img.save(f"{output_path}{self.__repr__()}.png")

    def calculate_column(self, x_val):
        """
        Is used by a particular process to compute a column of the image
        :param x_val: The column which the process is responsible for
        """

        column = np.array([(255, 255, 255)] * self.height)

        # sets up optimization using 180 degree rotational symmetry about the origin
        draw_range = range(self.height)  # Default
        if 0 < self.x_0 < self.width and 0 < self.y_0 < self.height:  # Is the origin in the image?
            if self.y_0 <= self.height / 2:  # Is the origin in the upper or lower half of the image?
                if self.x_0 <= self.width / 2:  # Is the origin in the left or right half of the image?
                    if self.x_0 < x_val <= 2*self.x_0-1:  # Is the particular x-value in the optimization region?
                        draw_range = [0] + list(range(2*self.y_0-1, self.height))

                else:
                    if x_val > self.x_0:  # Is the particular x-value in the optimization region?
                        draw_range = [0] + list(range(2*self.y_0, self.height))
            else:
                if self.x_0 <= self.width / 2:  # Is the origin in the left or right half of the image?
                    if self.x_0 < x_val <= 2*self.x_0-1:  # Is the particular x-value in the optimization region?
                        draw_range = range(0, self.y_0 // 2 + 1)

                else:
                    if x_val > self.x_0:  # Is the particular x-value in the optimization region?
                        draw_range = range(0, self.y_0 // 2 + 1)

        for y in draw_range:
            pt = (self.center.real - self.real_width / 2 + self.real_width / self.width * x_val
                  + 1j * (self.center.imag - self.imaginary_height / 2 + self.imaginary_height / self.height * y))

            color = self.compute_pixel(pt)
            column[y] = color

        return column


# Examples on how to use the Julia class
if __name__ == "__main__":
    import os
    try:
        os.mkdir("output")
    except FileExistsError:
        pass  # Output folder already exists, we are good to go.

    # Other initial complex numbers to try out
    c1 = -0.9-0.23j
    c2 = -0.226 + -1.01j
    c3 = 0.4+0.4j

    # An easy to compute Julia with the initial value = 0, should generate a perfect circle
    J1 = Julia(0+0j, Julia.log_gray, 200, 3, 3, (500, 500))
    timer_start = perf_counter()
    J1.generate(output_path="output", file_name="trivial_julia")
    timer_stop = perf_counter()
    print(f"Trivial Julia generated in {timer_stop - timer_start} seconds.")

    # Nothing special, I just find this cool
    J2 = Julia(-0.7854285296052694 + 0.1471995674073696j, Julia.exponential_coloring, 650, 0.15, 0.15, (10_000, 10_000))
    timer_start = perf_counter()
    J2.generate(output_path="output")
    timer_stop = perf_counter()
    print(f"J2 generated in {timer_stop - timer_start} seconds.")

    # Same deal as J2
    J3 = Julia(-0.6+0.45j, Julia.exponential_coloring, 500, 1, 1, (10_000, 10_000), center=0.5-0.25j)
    timer_start = perf_counter()
    J3.generate(output_path="output")
    timer_stop = perf_counter()
    print(f"J3 generated in {timer_stop - timer_start} seconds.")
