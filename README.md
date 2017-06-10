# OpenCL JPEG Encoder

Encodes an RGB image buffer to *JPEG* using *OpenCL*. Beside the final run length encoding, all previous steps such as color transformation, downsampling, discrete cosine transformation and quantification are performed on the *OpenCL* Device.

## Building
Beside the *OpenCL C++ Wrapper* and *make* there are no dependencies. The program can be built calling 
```
make
```

## Running 
The program encodes a raw ppm image to an jpeg image 
```
./jpeg_enc src.ppm out.jpg <quality>
```

## Usage 
```c++
/* Create the encoder */
jpeg::JPEGEncoder encoder(<cl_device_type>, <quality>);

/* Encode the image */
encoder.encode_image(<input_buffer>, <width>, <height>, <output_file>);
```

# Performance
The performance was measured running on a Radeon R9 290 encoding an image with 12079x7025 pixels showing a cove.
- Color conversion: 8.4ms
- Downsampling: 9.5ms
- DCT + Quantification: 20ms
- Time to copy buffers: 40ms
