# Class for Mandelbrot Set image generation

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


class Mandelbrot(Fractal):
    def __init__(self, coloring_scheme: Callable, iterations, real_width, imaginary_height,
                 image_dimensions: tuple[int, int], constant: complex = 0+0j, center: complex = 0+0j):
        """
        Constructs a Mandelbrot object
        :param coloring_scheme: A function that describes how to color the image based on the speed of divergence
        :type coloring_scheme: Callable[[int, int], tuple[int, int, int]]
        :param iterations: How many iterations of (z**2 + c) before assuming convergence
        :param real_width: The length of the real value interval
        :param imaginary_height: The length of the imaginary value interval
        :param image_dimensions: A tuple describing the pixel width and height of the image
        :param constant: The initial value, defaults to 0 for the classical Mandelbrot set
        :param center: The complex point at the center of the image, defaults to 0+0j
        """

        self.coloring_scheme: Callable[[int, int], tuple[int, int, int]] = coloring_scheme
        self.iterations = iterations
        self.real_width = real_width
        self.imaginary_height = imaginary_height
        self.constant = constant
        self.center = center

        self.dimensions = image_dimensions
        self.width = self.dimensions[0]
        self.height = self.dimensions[1]

    def __repr__(self):
        """
        Returns a string representation of the Mandelbrot object
        Is used as the default file name
        """
        return (f"Mandelbrot({self.coloring_scheme.__name__}, {self.iterations}, {self.real_width}, "
                f"{self.imaginary_height}, {self.dimensions}, constant={self.constant}, center={self.center})")

    def compute_pixel(self, point: complex):
        """
        Determines how many iterations a point requires before it is known to diverge (abs(p) > 2)
        Uses self.coloring_scheme to determine the right color for the point
        :param point: The complex point to be analyzed
        """

        value = self.constant
        for z in range(0, self.iterations):
            value = value ** 2 + point

            if abs(value) > 2:  # Diverges
                return self.coloring_scheme(z, self.iterations)

        return 0, 0, 0  # Converges

    def generate(self, output_path="", draw_center_circle=False, file_name=""):
        """
        Computes and saves the Mandelbrot image, prepares and starts many processes
        Builds a 2D array of RGB tuples that gets used to generate an image
        :param output_path: The file path to which the finished image will be saved
        :param draw_center_circle: Set this to True to draw a small circle around the image center
        :param file_name: Use this to override the default naming behavior, see Mandelbrot.__repr__
        """

        # Multi-process, uses tqdm to generate a progress bar + computation time estimation
        x_range = range(0, self.width)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = np.array(
                list(tqdm(executor.map(self.calculate_column, x_range), total=len(x_range))),
                dtype=np.uint8
            )

        # Single-process
        # results = np.array([self.draw(x) for x in range(self.width)], dtype=np.uint8)

        # transposes the width-height sub-matrix
        transposed_results = np.transpose(results, (1, 0, 2))
        img = Image.fromarray(transposed_results, 'RGB')

        # Center circle, included for debugging help
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

        # sets up optimization using symmetry about the real axis
        double_count = range(0, 0)
        single_count = range(0, self.height)

        # y of pixel where the real axis lies
        y_0 = self.height//2 - int(self.center.imag*self.height/self.imaginary_height)

        if abs(y_0) < self.height:

            if y_0 >= self.height / 2:
                double_count = range(y_0, self.height)
                single_count = range(0, 2*y_0-self.height+1)
            else:
                double_count = range(y_0, 2*y_0 + 1)
                single_count = range(2*y_0+1, self.height)

        for y in single_count:
            pt = (self.center.real - self.real_width / 2 + self.real_width / self.width * x_val
                  + 1j * (self.center.imag - self.imaginary_height / 2 + self.imaginary_height / self.height * y))

            color = self.compute_pixel(pt)
            column[y] = color

        for y in double_count:

            pt = (self.center.real - self.real_width / 2 + self.real_width / self.width * x_val
                  + 1j * (self.center.imag - self.imaginary_height / 2 + self.imaginary_height / self.height * y))

            color = self.compute_pixel(pt)
            column[y] = color
            column[2*y_0 - y] = color

        return column


# Examples on how to use the Mandelbrot class
if __name__ == "__main__":
    import os
    try:
        os.mkdir("output")
    except FileExistsError:
        pass  # Output folder already exists, we are good to go.

    # An easy to compute, low resolution, purple image of the entire Mandelbrot set
    low_res_mandelbrot = Mandelbrot(Mandelbrot.log_purple, 300, 4, 3, (1000, 750), center=-0.5+0j)
    timer_start = perf_counter()
    low_res_mandelbrot.generate(output_path="output", file_name="full_mandelbrot")
    timer_stop = perf_counter()
    print(f"Low resolution Mandelbrot generated in {timer_stop - timer_start} seconds.")

    # A computationally expensive, colorful, image of "Seahorse Valley"
    seahorse_valley = Mandelbrot(Mandelbrot.hsl_coloring, 500, 0.06, 0.08, (7500, 10000), center=-0.76 - 0.15j)
    timer_start = perf_counter()
    seahorse_valley.generate(output_path="output", draw_center_circle=True, file_name="seahorse_valley")
    timer_stop = perf_counter()
    print(f"Seahorse Valley generated in {timer_stop - timer_start} seconds.")

    # A mini-Mandelbrot within the image generated above, computed with 2,000 iterations
    mini_mandelbrot = Mandelbrot(Mandelbrot.hsl_coloring_continuous, 2000, 0.0015/2, 0.0009/2, (10_000, 6_000),
                                 center=-0.7387325763702393-0.16421866416931197j)
    timer_start = perf_counter()
    mini_mandelbrot.generate(output_path="output", draw_center_circle=True, file_name="mini_mandelbrot")
    timer_stop = perf_counter()
    print(f"Mini-Mandelbrot generated in {timer_stop - timer_start} seconds.")
