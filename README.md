# Fractals
Code to generate images of Mandelbrot and Julia sets

## Math Overview

For a good explanation on these fractals, see [this Numberphile video.](https://www.youtube.com/watch?v=FFftmWSzgmk) For a quick explanation, continue reading. These fractals are based on the convergence of the sequence

$$z_{n+1}=z_{n}^2+c$$

where $z$ and $c$ are complex numbers. Take any complex $c$. The Julia set is the set of points in the complex plane for which if this point is $z_0$, $z_{n+1}=z_{n}^2+c$ converges for some $c$. For example, choose $c = i$. We can see that $z_0=1$ is not in this Julia set.

$$z_1 = (1)^2 + i = 1 + i$$
$$z_2 = (1+i)^2 + i = 3i$$
$$z_3 = (3i)^2 + i = -9 + i$$
$$z_4 = (-9 + i)^2 + i = 80-17i$$
$$\vdots$$

If we continue to iterate, each successive $z$ will get farther from the origin. In fact, mathematicians have shown that if the absolute value of any $z_n$ is greater than or equal to $2$, then the point will diverge, regardless of the value of $c$. Since $|z_2| = |3i| = 3 \geq 2$, we could have come to this conclusion 2 steps earlier. This program colors each pixel based on how many iterations it takes before we are sure the point diverges. Though, we can only check a finite number. After this maximum number of iterations, we assume the point does not diverge and color it black.

The Mandelbrot set is done similarly. In this case, we start with $z_0 = 0$ and add the complex point. Thus, the Mandelbrot set includes every choice of $c$ for which $z_{n+1}=z_{n}^2+c$ does not diverge, rather than every choice of $z_0$ for the Julia set. Here is an example. We can see that $i$ is in the Mandelbrot set. After only a few iterations, we notice that the iterations enter a stable "orbit."

$$z_1 = (0)^2 + i = i$$
$$z_2 = (i)^2 + i = -1 + i$$
$$z_3 = (-1+i)^2 + i = -i$$
$$z_4 = (-i)^2 + i = -1 +i$$
$$z_5 = (-1+i)^2 + i = -i$$
$$\vdots$$

## Code Optimization

Running this calculation for every point in a large image is very expensive. So, I multiprocess the image calculation with `concurrent.futures.ProcessPoolExecutor` by dividing the image into arrays of tuples and mapping processes to each array. After the calculation, I combine the arrays into a 2D array of tuples which get turned into an image using PIL. Be warned, This program will likely use all the free computation power it can get its hands on.

Mandelbrot sets are symmetric about the real axis (this is the line of points with an imaginary component of 0). So, when computing an array of points, I find which region I don't have to calculate, I don't compute it, but instead use the color of the corresponding point at its reflection once it is computed.

Julia sets have a $\pi$ radians rotational symmetry about the origin. Turn the set upside down and you'll have the same set. So, I find the region I don't have to calculate, wait until after the whole rest of the 2D array is computed, then copy the corresponding region, rotate it, and slide it in place.
