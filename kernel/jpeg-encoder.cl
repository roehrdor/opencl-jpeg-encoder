#define RED_OFFSET 0x0
#define GREEN_OFFSET 0x300
#define BLUE_OFFSET 0x600

__kernel void color_space_transform(__global unsigned int *color_conversion_table,
									__global unsigned char *image, unsigned int sz)
{
	size_t gx = get_global_id(0);

	/* only execute inside of the image range */
	if(gx < sz)
	{
		gx *= 3;

		/* read RGB values */
		unsigned char r = image[gx + 0];
		unsigned char g = image[gx + 1];
		unsigned char b = image[gx + 2];

		/* convert them into yCbCr */
		unsigned int ry = color_conversion_table[0 + r * 3 + 0];
		unsigned int rcr = color_conversion_table[0 + r * 3 + 1];
		unsigned int rcb = color_conversion_table[0 + r * 3 + 2];

		unsigned int gy = color_conversion_table[GREEN_OFFSET + g * 3 + 0];
		unsigned int gcr = color_conversion_table[GREEN_OFFSET + g * 3 + 1];
		unsigned int gcb = color_conversion_table[GREEN_OFFSET + g * 3 + 2];

		unsigned int by = color_conversion_table[BLUE_OFFSET + b * 3 + 0];
		unsigned int bcr = color_conversion_table[BLUE_OFFSET + b * 3 + 1];
		unsigned int bcb = color_conversion_table[BLUE_OFFSET + b * 3 + 2];

		/* store them back */
		image[gx + 0] = ((unsigned char)((ry + gy + by) >> 0x10));
		image[gx + 1] = ((unsigned char)((rcb + gcb + bcb) >> 0x10));
		image[gx + 2] = ((unsigned char)((rcr + gcr + bcr) >> 0x10));
	}
}


__kernel void downsample_full(__global short *buffer, __global unsigned char *image,
							  unsigned int nsbw, unsigned int nbw,
							  unsigned int nbh, unsigned int width, unsigned int height)
{
	size_t gx = get_global_id(0);

	/* compute id of super block */
	size_t super_block_id = gx >> 0x8;

	/* compute x and y of super block */
	size_t super_block_x = super_block_id % nsbw;
	size_t super_block_y = super_block_id / nsbw;

	/* super sub block id and x and y position */
	size_t sub_block_id = (gx & 0xFF) >> 0x6;
	size_t sub_block_x = sub_block_id & 0x1;
	size_t sub_block_y = sub_block_id >> 0x1;

	/* compute in block index and x and y index */
	size_t field_id = gx & 0x3F;
	size_t field_x = field_id & 0x7;
	size_t field_y = field_id >> 0x3;

	/* Global x and y image position */
	size_t image_x = (super_block_x << 0x4) | (sub_block_x << 0x3) | field_x;
	size_t image_y = (super_block_y << 0x4) | (sub_block_y << 0x3) | field_y;

	/* Clamp */
	if(image_x >= width) image_x = width - 1;
	if(image_y >= height) image_y = height - 1;

	/* Copy the pixel */
	buffer[gx] = (short)image[(image_x + (image_y * width)) * 3] - (short)0x80;
}

__kernel void downsample_2v2(__global short *cb, __global short *cr,
							 __global unsigned char *image, unsigned int nsbw,
							 unsigned int nbw, unsigned int nbh,
							 unsigned int width, unsigned int height)
{
	size_t gx = get_global_id(0);

	/* compute id of super block */
	size_t super_block_id = gx >> 0x6;		/* divide by 64 */

	/* compute x and y of super block */
	size_t super_block_x = super_block_id % nsbw;
	size_t super_block_y = super_block_id / nsbw;

	/* super sub block id and x and y position */
	size_t sub_block_x = (gx & 0x7) > 0x3;
	size_t sub_block_y = (gx & 0x3F) > 0x1F;

	/* compute in block index and x and y index */
	size_t field_id = gx & 0x3F;
	size_t field_x = (field_id & 0x7) << 0x1;
	size_t field_y = (field_id >> 0x3) << 0x1;

	/* Global x and y image position */
	size_t image_x = (super_block_x << 0x4) | (sub_block_x << 0x3) | field_x;
	size_t image_y = (super_block_y << 0x4) | (sub_block_y << 0x3) | field_y;

	/* Compute pixels x and y values */
	size_t pixel_x0 = image_x;
	size_t pixel_x1 = image_x + 1;
	size_t pixel_y0 = image_y;
	size_t pixel_y1 = image_y + 1;

	/* Clamp */
	if(pixel_x0 >= width) pixel_x0 = width - 1;
	if(pixel_x1 >= width) pixel_x1 = width - 1;
	if(pixel_y0 >= height) pixel_y0 = height - 1;
	if(pixel_y1 >= height) pixel_y1 = height - 1;

	/* compute pixel ids */
	size_t pixel00 = (pixel_x0 + (pixel_y0 * width));
	size_t pixel10 = (pixel_x1 + (pixel_y0 * width));
	size_t pixel01 = (pixel_x0 + (pixel_y1 * width));
	size_t pixel11 = (pixel_x1 + (pixel_y1 * width));

	/* Sum up the components */
	long cb_sum = 0;
	long cr_sum = 0;

	size_t pixel = pixel00 * 3;
	cb_sum += (long)image[pixel + 1];
	cr_sum += (long)image[pixel + 2];

	pixel = pixel10 * 3;
	cb_sum += (long)image[pixel + 1];
	cr_sum += (long)image[pixel + 2];

	pixel = pixel01 * 3;
	cb_sum += (long)image[pixel + 1];
	cr_sum += (long)image[pixel + 2];

	pixel = pixel11 * 3;
	cb_sum += (long)image[pixel + 1];
	cr_sum += (long)image[pixel + 2];

	int bias = 0x1 << (gx & 0x1);
	cb_sum += bias;
	cr_sum += bias;

	/* Store the result */
	cb[gx] = (short)(cb_sum >> 0x2) - (short)0x80;
	cr[gx] = (short)(cr_sum >> 0x2) - (short)0x80;
}


/*
 *  NOTE: this algorithm is described in C. Loeffler, A. Ligtenberg and G. Moschytz, "Practical Fast 1-D DCT
 *   Algorithms with 11 Multiplications", Proc. Int'l. Conf. on Acoustics,
 *   Speech, and Signal Processing 1989 (ICASSP '89), pp. 988-991.
 */
#define LEFT_SHIFT(a, b) ((int)((unsigned int)(a) << (b)))
#define DESCALE(x,n)  RIGHT_SHIFT((x) + (1 << ((n)-1)), n)
#define RIGHT_SHIFT(x,shft)     ((x) >> (shft))
__kernel void dct_quant(__global short *block, __global short *divisors, unsigned int divisor_offset,
						__global short *multiplier, __global int *sign, __global int *indices,
						__global char *descaler, __global short *descaler_offset)
{
	unsigned int product;
	unsigned short recip, corr;
	short ioffset, moffset, soffset, doffset;
	short t0, t1, t2, t3, res, neg;
	int value;
	__local short *dataptr;
	int shift;

	size_t gx = get_global_id(0);
	size_t lx = get_local_id(0);

	short row = lx >> 0x3;
	short row_offset = (row) << 0x3;
	short column = lx & 0x7;

	__local short lblock[0x40];
	lblock[lx] = block[gx];
	barrier(CLK_LOCAL_MEM_FENCE);
	dataptr = &lblock[row_offset];

	/* Pass 1: process rows. */
	ioffset = column << 0x3;
	moffset = column << 0x2;
	soffset = column << 0x1;
	doffset = column << 0x1;
	t0 = dataptr[indices[ioffset + 0]] + (dataptr[indices[ioffset + 1]] * sign[soffset + 0]);
	t1 = dataptr[indices[ioffset + 2]] + (dataptr[indices[ioffset + 3]] * sign[soffset + 0]);
	t2 = dataptr[indices[ioffset + 4]] + (dataptr[indices[ioffset + 5]] * sign[soffset + 0]);
	t3 = dataptr[indices[ioffset + 6]] + (dataptr[indices[ioffset + 7]] * sign[soffset + 0]);
	value = t0 * multiplier[moffset + 0] + (t1 + t0) * multiplier[moffset + 1] + (t2 + t0)
			   * multiplier[moffset + 2] + ((t0 + t1) + ((t2 + t3) * sign[soffset + 1])) * multiplier[moffset + 3];
	res = (short)DESCALE(value, 0xB) * descaler[doffset + 0] +	LEFT_SHIFT(value, 0x2) * descaler[doffset + 1];

	/* Wait for all rows in the local execution to complete */
	barrier(CLK_LOCAL_MEM_FENCE);
	lblock[lx] = res;
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Pass 2: process columns */
	dataptr = &lblock[column];

	ioffset = row << 0x3;
	moffset = row << 0x2;
	soffset = row << 0x1;
	t0 = dataptr[indices[ioffset + 0] << 0x3] + (dataptr[indices[ioffset + 1] << 0x3] * sign[soffset + 0]);
	t1 = dataptr[indices[ioffset + 2] << 0x3] + (dataptr[indices[ioffset + 3] << 0x3] * sign[soffset + 0]);
	t2 = dataptr[indices[ioffset + 4] << 0x3] + (dataptr[indices[ioffset + 5] << 0x3] * sign[soffset + 0]);
	t3 = dataptr[indices[ioffset + 6] << 0x3] + (dataptr[indices[ioffset + 7] << 0x3] * sign[soffset + 0]);
	value = t0 * multiplier[moffset + 0] + (t1 + t0) * multiplier[moffset + 1] + (t2 + t0)
			   * multiplier[moffset + 2] + ((t0 + t1) + ((t2 + t3) * sign[soffset + 1])) * multiplier[moffset + 3];
	res = DESCALE(value, 0x2 + descaler_offset[row]);

	/* Pass 3: quantize */
	recip = divisors[divisor_offset + lx + 0x40 * 0];
	corr = divisors[divisor_offset + lx + 0x40 * 1];
	shift = divisors[divisor_offset + lx + 0x40 * 3];
	neg = res < 0 ? -1 : 1;
	res *= neg;
	product = (unsigned int) (res + corr) * recip;
	product >>= shift + sizeof(short) * 8;
	res = (short) product;
	res *= neg;
	block[gx] = (short)res;
}

__kernel void zero_out_right(__global short *buffer, unsigned int nsbw, unsigned int nsbh, unsigned int nbw)
{
	size_t gx = get_global_id(0);
	size_t super_block_x = nsbw - 1;
	if ((super_block_x << 0x1) + 1 >= nbw) {
		size_t super_block_y = gx >> 0x7;
		size_t super_block_id = (super_block_y * nsbw) + super_block_x;
		size_t local_block_id = (gx & 0x7F) > 0x3F ? 3 : 1;
		size_t field_id = (gx & 0x3F);
		size_t block_id = ((super_block_id << 0x8) | (local_block_id << 0x6) | field_id);
		if (field_id == 0)
		{
			size_t left_block_0_id = ((super_block_id << 0x8)| ((local_block_id - 1) << 0x6) | field_id);
			buffer[block_id] = buffer[left_block_0_id];
		}
		else
		{
			buffer[block_id] = 0;
		}
	}
}

__kernel void zero_out_bottom(__global short *buffer, unsigned int nsbw, unsigned int nsbh, unsigned int nbh)
{
	size_t gx = get_global_id(0);
	size_t super_block_y = nsbh - 1;
	if ((super_block_y << 0x1) + 1 >= nbh) {
		size_t super_block_x = gx >> 0x7;
		size_t super_block_id = (super_block_y * nsbw) + super_block_x;
		size_t local_block_id = (gx & 0x7F) > 0x3F ? 3 : 2;
		size_t field_id = (gx & 0x3F);
		size_t block_id = ((super_block_id << 0x8) | (local_block_id << 0x6) | field_id);

		/* Note we do NOT copy the value from the neighbor field, since the value in the neighbor
		 * field is not yet guaranteed to be entered. Therefore fill out with zeroes and copy single
		 * values back on the host before performing entropy encoding */
		buffer[block_id] = 0;
	}
}

