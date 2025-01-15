# Utilities

import numpy as np


# HSL to RGB conversion
def hsl_to_rgb(h, s, l):
    c = (1 - abs(2*l-1)) * s
    x = c * (1 - abs((h/60) % 2 - 1))
    m = l - c/2

    pre_rgb = (0, 0, 0)
    if 0 <= h < 60:
        pre_rgb = (c, x, 0)
    elif 60 <= h < 120:
        pre_rgb = (x, c, 0)
    elif 120 <= h < 180:
        pre_rgb = (0, c, x)
    elif 180 <= h < 240:
        pre_rgb = (0, x, c)
    elif 240 <= h < 300:
        pre_rgb = (x, 0, c)
    elif 300 <= h < 360:
        pre_rgb = (c, 0, x)

    return round((pre_rgb[0]+m) * 255), round((pre_rgb[1]+m) * 255), round((pre_rgb[2]+m) * 255)


class Fractal:
    """
    Parent class for Mandelbrot and Julia
    Organizes coloring functions as static methods
    """
    @staticmethod
    def exponential_coloring(iterations_before_diverged, max_iterations):
        """
        Uses a series of Gaussian and logistic curves to smoothly transition from black, to red, to green, to sky blue,
        to dark blue; the red value depends on an increasing concave-down parabola until the maximum of said parabola,
        where it smoothly switches to a Gaussian. The green value depends on a gaussian. The blue value depends on a
        logistic.
        """

        r: float
        if iterations_before_diverged <= max_iterations / 4:
            r = (-4080 / max_iterations ** 2 * iterations_before_diverged ** 2
                 + 2040 / max_iterations * iterations_before_diverged)
        else:
            r = 255 * np.exp(-((iterations_before_diverged - 0.25 * max_iterations) / (0.1 * max_iterations)) ** 2)

        g = 255 * np.exp(-((iterations_before_diverged - 0.5 * max_iterations) / (0.25 * max_iterations)) ** 2)

        b = 255 / (1 + np.exp(-10 / max_iterations * (iterations_before_diverged - max_iterations / 2)))

        return round(r), round(g), round(b)

    @staticmethod
    def hsl_coloring(iterations_before_diverged, max_iterations) -> tuple[int, int, int]:
        """
        Uses the rate of divergence to place a color along the HSL color wheel, then converts to RGB
        The outputted color is always on the circle with 100% saturation and
        Repeats every max_iterations worth of iterations
        """

        hue = 360 * iterations_before_diverged / max_iterations

        return hsl_to_rgb(hue, 1, 0.5)

    @staticmethod
    def hsl_coloring_continuous(iterations_before_diverged, _):
        """
        Similar to hsl_coloring, but traverses more colors and repeats every 720 iterations
        :param _: A dummy argument to avoid runtime errors, Mandelbrot and Julia expect two arguments
        """

        radians = iterations_before_diverged * 2 * np.pi / 360 + 2 * np.pi
        lightness = (0.75 + 1.75 * (np.cos(1.75 * radians)) ** 2) / 3

        return hsl_to_rgb(iterations_before_diverged % 360, 1, lightness)

    @staticmethod
    def log_gray(iterations_before_diverged, max_iterations):
        """
        Uses a logarithm to smooth out color transition at the fractal boundary
        shade = 255 * log_{Base: max_iterations}(iterations_before_diverged + 1)
        """

        # Prevents division by 0, log(1) = 0
        if max_iterations == 1:
            return (0, 0, 0) if iterations_before_diverged else (255, 255, 255)

        shade_of_gray = int(
            255 * np.log(iterations_before_diverged + 1) / np.log(max_iterations))  # Logarithm change of base
        return shade_of_gray, shade_of_gray, shade_of_gray

    @staticmethod
    def log_purple(iterations_before_diverged, max_iterations):
        """
        Similar to log_grayscale, but interpolates (0,0,0) to (190, 3, 252),
        or from black, through dark purple, to magenta
        """

        # Prevents division by 0, log(1) = 0
        if max_iterations == 1:
            return (0, 0, 0) if iterations_before_diverged else (190, 3, 252)

        scalar = np.log(iterations_before_diverged + 1) / np.log(max_iterations)  # Logarithm change of base
        return int(190 * scalar), int(3 * scalar), int(252 * scalar)
