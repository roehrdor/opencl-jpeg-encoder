#include "../include/jpeg_encoder.hpp"

namespace jpeg
{

#define SCALEBITS 0x10
#define FIX(x)          ((unsigned int) ((x) * (1L<<SCALEBITS) + 0.5))
#define ONE_HALF        ((unsigned int) 1 << (SCALEBITS-1))
#define CBCR_OFFSET     ((unsigned int) 0x80 << SCALEBITS)

/**
 * Push back the given the value to the output buffer (single byte only)
 *
 * @param output_buf the buffer
 * @param value the value
 */
static void write_byte(std::vector<char>& output_buf, int value)
{
	output_buf.push_back((char)value);
}

/**
 * Write two bytes to the output buffer by splitting it and writing those two bytes
 * seperately
 *
 * @param output_buf the buffer
 * @param value the value
 */
static void write_2byte(std::vector<char>& output_buf, int value)
{
	write_byte(output_buf, (value >> 0x8) & 0xFF);
	write_byte(output_buf, value & 0xFF);
}

/**
 * Export a JPEG marker
 *
 * @param output_buf the buffer
 * @param value the marker
 */
static void write_marker(std::vector<char>& output_buf, int value)
{
	write_byte(output_buf, 0xFF);
	write_byte(output_buf, value);
}

/**
 * Compute the reciprocal for the divisor and save in the given table
 * the reciprocal, length and shift values
 * Taken from: https://github.com/libjpeg-turbo/libjpeg-turbo/
 *
 * @param divisor the divisor
 * @param dtbl table to store
 */
static int compute_reciprocal (unsigned short divisor, short *dtbl)
{
	unsigned int fq, fr;
	unsigned short c;
	int b, r;

	if (divisor == 1)
	{
		dtbl[0x40 * 0] = (short) 1;						/* reciprocal */
		dtbl[0x40 * 1] = (short) 0;						/* correction */
		dtbl[0x40 * 2] = (short) 1;						/* scale */
		dtbl[0x40 * 3] = -(short) (sizeof(short) * 8);	/* shift */
		return 0;
	}

	b = nbits_table[divisor] - 1;
	r = sizeof(short) * 8 + b;

	fq = ((unsigned int)1 << r) / divisor;
	fr = ((unsigned int)1 << r) % divisor;

	c = divisor >> 0x1;

	if (fr == 0)
	{
		fq >>= 1;
		r--;
	}
	else if (fr <= (divisor / 2U))
	{
		c++;
	}
	else
	{
		fq++;
	}

	dtbl[0x40 * 0] = (short) fq;
	dtbl[0x40 * 1] = (short) c;
	dtbl[0x40 * 2] = (short) (1 << (sizeof(short)*8*2 - r));
	dtbl[0x40 * 3] = (short) r - sizeof(short)*8;

	return r <= 16 ? 0 : 1;
}

/**
 * Build the program from the given file
 *
 * @param context the OpenCL context to build the program in
 * @param device the device to build for
 * @param file the kernel file
 * @return the created program
 */
static cl::Program build_from_file(cl::Context &context, cl::Device &device, const char* const file)
{
	std::ifstream t(file);
	std::string str;

	t.seekg(0, std::ios::end);
	str.reserve(t.tellg());
	t.seekg(0, std::ios::beg);

	str.assign((std::istreambuf_iterator<char>(t)),
	            std::istreambuf_iterator<char>());
	cl::Program ret(context, str);
	ret.build({device});
	return ret;
}

/**
 * Create a new encoder
 *
 * @param type the device type to use
 * @param quality the quality setting to use (clamped between 1 and 100)
 */
JPEGEncoder::JPEGEncoder(cl_device_type type, unsigned char quality) :
		m_context(type),
		m_device(m_context.getInfo<CL_CONTEXT_DEVICES>()[0]),
		m_queue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE),
		m_program(build_from_file(m_context, m_device, "kernel/jpeg-encoder.cl")),
		md_color_conversion_table(m_context, CL_MEM_READ_ONLY, sizeof(color_conversion_table)),
		md_fdct_divisors(m_context, CL_MEM_READ_ONLY, sizeof(m_fdct_divisors)),
		md_fdct_multiplier(m_context, CL_MEM_READ_ONLY, sizeof(MULTIPLIER)),
		md_fdct_sign(m_context, CL_MEM_READ_ONLY, sizeof(SIGN)),
		md_fdct_indices(m_context, CL_MEM_READ_ONLY, sizeof(INDICES)),
		md_fdct_descaler(m_context, CL_MEM_READ_ONLY, sizeof(DESCALER)),
		md_fdct_descaler_offset(m_context, CL_MEM_READ_ONLY, sizeof(DESCALER_OFFSET))
{
	this->create_encoder(quality);
	this->prepare_device();
}

/**
 * Prepare the device, create kernels and write conversion table and divisor table to device
 */
void JPEGEncoder::prepare_device(void)
{
	/* copy tables to device */
	this->m_queue.enqueueWriteBuffer(this->md_color_conversion_table, false, 0, sizeof(color_conversion_table), color_conversion_table);
	this->m_queue.enqueueWriteBuffer(this->md_fdct_divisors, false, 0, sizeof(m_fdct_divisors), &this->m_fdct_divisors);
	this->m_queue.enqueueWriteBuffer(this->md_fdct_multiplier, false, 0, sizeof(MULTIPLIER), &MULTIPLIER);
	this->m_queue.enqueueWriteBuffer(this->md_fdct_sign, false, 0, sizeof(SIGN), &SIGN);
	this->m_queue.enqueueWriteBuffer(this->md_fdct_indices, false, 0, sizeof(INDICES), &INDICES);
	this->m_queue.enqueueWriteBuffer(this->md_fdct_descaler, false, 0, sizeof(DESCALER), &DESCALER);
	this->m_queue.enqueueWriteBuffer(this->md_fdct_descaler_offset, false, 0, sizeof(DESCALER_OFFSET), &DESCALER_OFFSET);

	/* create kernels */
	this->m_transformation_kernel = cl::Kernel(this->m_program, "color_space_transform");
	this->m_downsample_full_kernel = cl::Kernel(this->m_program, "downsample_full");
	this->m_downsample_2v2_kernel = cl::Kernel(this->m_program, "downsample_2v2");
	this->m_dct_quant = cl::Kernel(this->m_program, "dct_quant");
	this->m_zero_out_right = cl::Kernel(this->m_program, "zero_out_right");
	this->m_zero_out_bottom = cl::Kernel(this->m_program, "zero_out_bottom");
}

/**
 * Encode the given image
 *
 * @param image pointer to the image data in flat row major layout
 * @param width of the image
 * @param height of the image
 * @param file the output file to store the image at
 * @return 0 on success
 */
int JPEGEncoder::encode_image(unsigned char *image, size_t width, size_t height, const char * const file)
{
	size_t wg;
	std::vector<char> output_buffer;
	FILE *fp;

	/* Make sure the image pointer is valid */
	if(image == NULL)
	{
		fprintf(stderr, "Image data needs to be provided\n");
		return 0x2;
	}

	/* Validate file handler */
	fp = fopen(file, "wb");
	if(fp == NULL)
	{
		fprintf(stderr, "The file \'%s\' could not be opened, aborting compressing\n", file);
		return 0x1;
	}

	/* Write the file, frame and scan header to the output buffer */
	this->write_file_header(output_buffer);
	this->write_frame_header(output_buffer, width, height);
	this->write_scan_header(output_buffer);

	//
	// Color Space Transformation
	//
	/* Initialize image buffer */
	cl::Buffer image_buffer(this->m_context, CL_MEM_READ_WRITE, sizeof(unsigned char) * 3 * width * height);
	this->m_queue.enqueueWriteBuffer(image_buffer, true, 0, sizeof(unsigned char) * 3 * width * height, image);

	/* Set arguments */
	this->m_transformation_kernel.setArg<cl::Buffer>(0, this->md_color_conversion_table);
	this->m_transformation_kernel.setArg<cl::Buffer>(1, image_buffer);
	this->m_transformation_kernel.setArg<cl_uint>(2, (cl_uint)(width * height));

	/* Compute work group size to be the closest bigger multiple of 64 to the number of pixels in the image */
	wg = (((width * height) + 0x3F) >> 0x6) << 0x6;
	this->m_queue.enqueueNDRangeKernel(this->m_transformation_kernel, 0, wg, 0x40);


	//
	// Downsampling
	//
	/* Do the downsampling for the y channel
	 * This does a full downsample, which means keeping all the pixels we already have
	 * For future processing the downsampling splits the image, which is currently in flat
	 * row layout into super blocks containing of 4 sub blocks each which represent a full
	 * MCU block
	 *
	 * Where each rx is a row and each ax/bx/.../zx is a super block
	 * 										+---------
	 *										| a1 a2 a3 ...
	 *	[r1.....][r2.....]....[rn.....] =>	| b1 b2 b3 ...
	 *										..............
	 *										| z1 z2 z3 ...
	 *										+---------			The super block is stored in
	 *											|				in flat layout hierarchy
	 *											v
	 *						[a1 a2 a3...][b1 b2 b3...]...[z1 z2 z3...]
	 *
	 * [0][1][2][3]
	 *	   ^
	 *	   |
	 *  +-----+		Each superblock contains 4 sub blocks which are
	 *	| 0 1 |		are ordered as displayed, where each of the four
	 *	| 2 3 |		sub blocks represents a full MCU block
	 *	+-----+		the are stored in a flat layout in memory
	 */

	/* Compute the number of blocks in x and y direction */
	cl_uint nbw = (width + 0x7) >> 0x3;
	cl_uint nbh = (height + 0x7) >> 0x3;

	/* Compute the number of super blocks in x and y direction */
	cl_uint nsbw = (width + 0xF) >> 0x4;
	cl_uint nsbh = (height + 0xF) >> 0x4;

	/* Compute work group size */
	wg = (nsbw * nsbh) << 0x8;

	/* Initialize the block buffer */
	cl::Buffer y_block_buffer(this->m_context, CL_MEM_READ_WRITE, wg * sizeof(cl_short));

	/* Set the kernel arguments */
	this->m_downsample_full_kernel.setArg<cl::Buffer>(0, y_block_buffer);
	this->m_downsample_full_kernel.setArg<cl::Buffer>(1, image_buffer);
	this->m_downsample_full_kernel.setArg<cl_uint>(2, nsbw);
	this->m_downsample_full_kernel.setArg<cl_uint>(3, nbw);
	this->m_downsample_full_kernel.setArg<cl_uint>(4, nbh);
	this->m_downsample_full_kernel.setArg<cl_uint>(5, (cl_uint)width);
	this->m_downsample_full_kernel.setArg<cl_uint>(6, (cl_uint)height);

	/* Execute kernel */
	this->m_queue.enqueueNDRangeKernel(this->m_downsample_full_kernel, 0, wg, 0x40);


	/* Downsample Cb/Cr Channels */
	/* The number of blocks and super blocks stays the same,
	 * since we do a 2:2 downsample only a fourth of the number
	 * of original items are stored. */
	wg = (nsbw * nsbh) << 0x6;

	/* Create buffer for the cb and cr channels */
	cl::Buffer cb_block_buffer(this->m_context, CL_MEM_READ_WRITE, wg * sizeof(cl_short));
	cl::Buffer cr_block_buffer(this->m_context, CL_MEM_READ_WRITE, wg * sizeof(cl_short));

	/* Set the kernel arguments */
	this->m_downsample_2v2_kernel.setArg<cl::Buffer>(0, cb_block_buffer);
	this->m_downsample_2v2_kernel.setArg<cl::Buffer>(1, cr_block_buffer);
	this->m_downsample_2v2_kernel.setArg<cl::Buffer>(2, image_buffer);
	this->m_downsample_2v2_kernel.setArg<cl_uint>(3, nsbw);
	this->m_downsample_2v2_kernel.setArg<cl_uint>(4, nbw);
	this->m_downsample_2v2_kernel.setArg<cl_uint>(5, nbh);
	this->m_downsample_2v2_kernel.setArg<cl_uint>(6, (cl_uint) width);
	this->m_downsample_2v2_kernel.setArg<cl_uint>(7, (cl_uint) height);

	/* Execute the kernel */
	this->m_queue.enqueueNDRangeKernel(this->m_downsample_2v2_kernel, 0, wg, 0x40);

	//
	// DCT and Quantification
	//
	/* Prepare and execute kernel for y channel */
	wg = (nsbw * nsbh) << 0x8;
	this->m_dct_quant.setArg<cl::Buffer>(0, y_block_buffer);
	this->m_dct_quant.setArg<cl::Buffer>(1, this->md_fdct_divisors);
	this->m_dct_quant.setArg<cl_uint>(2, 0);
	this->m_dct_quant.setArg<cl::Buffer>(3, this->md_fdct_multiplier);
	this->m_dct_quant.setArg<cl::Buffer>(4, this->md_fdct_sign);
	this->m_dct_quant.setArg<cl::Buffer>(5, this->md_fdct_indices);
	this->m_dct_quant.setArg<cl::Buffer>(6, this->md_fdct_descaler);
	this->m_dct_quant.setArg<cl::Buffer>(7, this->md_fdct_descaler_offset);
	this->m_queue.enqueueNDRangeKernel(this->m_dct_quant, 0x0, wg, 0x40);

	/* Prepare and execute kernel for cb channel */
	wg = (nsbw * nsbh) << 0x6;
	this->m_dct_quant.setArg<cl::Buffer>(0, cb_block_buffer);
	this->m_dct_quant.setArg<cl::Buffer>(1, this->md_fdct_divisors);
	this->m_dct_quant.setArg<cl_uint>(2, 0x100);
	this->m_dct_quant.setArg<cl::Buffer>(3, this->md_fdct_multiplier);
	this->m_dct_quant.setArg<cl::Buffer>(4, this->md_fdct_sign);
	this->m_dct_quant.setArg<cl::Buffer>(5, this->md_fdct_indices);
	this->m_dct_quant.setArg<cl::Buffer>(6, this->md_fdct_descaler);
	this->m_dct_quant.setArg<cl::Buffer>(7, this->md_fdct_descaler_offset);
	this->m_queue.enqueueNDRangeKernel(this->m_dct_quant, 0x0, wg, 0x40);

	/* Prepare and execute kernel for cr channel */
	this->m_dct_quant.setArg<cl::Buffer>(0, cr_block_buffer);
	this->m_dct_quant.setArg<cl::Buffer>(1, this->md_fdct_divisors);
	this->m_dct_quant.setArg<cl_uint>(2, 0x100);
	this->m_dct_quant.setArg<cl::Buffer>(3, this->md_fdct_multiplier);
	this->m_dct_quant.setArg<cl::Buffer>(4, this->md_fdct_sign);
	this->m_dct_quant.setArg<cl::Buffer>(5, this->md_fdct_indices);
	this->m_dct_quant.setArg<cl::Buffer>(6, this->md_fdct_descaler);
	this->m_dct_quant.setArg<cl::Buffer>(7, this->md_fdct_descaler_offset);
	this->m_queue.enqueueNDRangeKernel(this->m_dct_quant, 0x0, wg, 0x40);

	/* Zero out unused blocks on the right side */
	wg = (nbh << 0x6);
	this->m_zero_out_right.setArg<cl::Buffer>(0, y_block_buffer);
	this->m_zero_out_right.setArg<cl_uint>(1, (cl_uint)nsbw);
	this->m_zero_out_right.setArg<cl_uint>(2, (cl_uint)nsbh);
	this->m_zero_out_right.setArg<cl_uint>(3, (cl_uint)nbw);
	this->m_queue.enqueueNDRangeKernel(this->m_zero_out_right, 0, wg, 0x40);

	/* Zero out unsued blocks on the bottom of the image */
	wg = (nsbw << 0x7);
	this->m_zero_out_bottom.setArg<cl::Buffer>(0, y_block_buffer);
	this->m_zero_out_bottom.setArg<cl_uint>(1, (cl_uint)nsbw);
	this->m_zero_out_bottom.setArg<cl_uint>(2, (cl_uint)nsbh);
	this->m_zero_out_bottom.setArg<cl_uint>(3, (cl_uint)nbh);
	this->m_queue.enqueueNDRangeKernel(this->m_zero_out_bottom, 0, wg, 0x80);

	/* Copy result back to host to perform entropy on host device */
	short *y_buffer = (short*)malloc(sizeof(short) * (nsbw * nsbh) << 0x8);
	short *cb_buffer = (short*)malloc(sizeof(short) * (nsbw * nsbh) << 0x6);
	short *cr_buffer = (short*)malloc(sizeof(short) * (nsbw * nsbh) << 0x6);
	this->m_queue.enqueueReadBuffer(y_block_buffer, true, 0, sizeof(short) * (nsbw * nsbh) << 0x8, y_buffer);
	this->m_queue.enqueueReadBuffer(cb_block_buffer, true, 0, sizeof(short) * (nsbw * nsbh) << 0x6, cb_buffer);
	this->m_queue.enqueueReadBuffer(cr_block_buffer, true, 0, sizeof(short) * (nsbw * nsbh) << 0x6, cr_buffer);

	/* For convenient access, cast to 3D/2D arrays */
	short (*y_blocks)[0x4][0x40] = (short (*)[0x4][0x40])y_buffer;
	short (*cb_blocks)[0x40] = (short (*)[0x40])cb_buffer;
	short (*cr_blocks)[0x40] = (short (*)[0x40])cr_buffer;

	/* As mentioned in the kernel code we can not the field zero values in the kernel
	 * since neighboring blocks are processed concurrently.
	 * Since the entropy encoding is performed on the host the data needs to be copied
	 * anyways, so setting these values on the host does not introduce an extra
	 * copy operation */
	size_t super_block_y = nsbh - 1;
	size_t super_block_id_base = (super_block_y * nsbw);
	for(size_t gx = 0; gx < nsbw; ++gx)
	{
		if ((super_block_y << 0x1) + 1 >= nbh) {
			size_t super_block_x = gx;
			size_t super_block_id = super_block_id_base + super_block_x;
			short value = y_blocks[super_block_id][1][0];
			y_blocks[super_block_id][2][0] = value;
			y_blocks[super_block_id][3][0] = value;
		}
	}

	//
	// Entropy coding
	//
	wg = (nsbw * nsbh);		/* number of super blocks */
	short *mcu_buffer[0x6];
	entropy_state_t state;
	memset(state.last_dc_val, 0, sizeof(state.last_dc_val));
	state.bits = 0;
	for(size_t i = 0; i < wg; ++i)
	{
		/* Perform entropy encoding on each block */
		mcu_buffer[0] = y_blocks[i][0];
		mcu_buffer[1] = y_blocks[i][1];
		mcu_buffer[2] = y_blocks[i][2];
		mcu_buffer[3] = y_blocks[i][3];
		mcu_buffer[4] = cb_blocks[i];
		mcu_buffer[5] = cr_blocks[i];
		this->encode_entropy(mcu_buffer, output_buffer, state);
	}


	/* Flush Entropy */
	size_t bits = state.bits;
	size_t buffer = state.buffer;
	bits += 7;
	buffer = (buffer << 0x7) | 0x7F;
	while(bits > 0x7)
	{
		bits -= 0x8;
		unsigned char c = (unsigned char)(buffer >> bits);
		output_buffer.push_back(c);
		if(c == 0xFF)
			output_buffer.push_back(c);
	}

	/* Write the file tailor to the output buffer */
	write_marker(output_buffer, 0xD9);

	/* write the content to file */
	(void)fwrite(output_buffer.data(), sizeof(char),  output_buffer.size(), fp);
	fclose(fp);

	/* Release allocated memory */
	free(y_buffer);
	free(cb_buffer);
	free(cr_buffer);

	return 0x0;
}

/**
 * Write the file header
 *
 * @param output_buf the output buffer to use
 */
void JPEGEncoder::write_file_header(std::vector<char>& output)
{
	static unsigned char headMagic[] = {0xFF, 0xD8, 0xFF, 0xE0};
	static unsigned char jfifApp0[] = {0x00, 0x10, 'J', 'F', 'I', 'F', 0x0, 0x1, 0x1, 0x0, 0x0, 0x1, 0x0, 0x1, 0x0, 0x0};

	/* Copy the head to the output buffer */
	for(size_t i = 0; i < 0x4; ++i)
	{
		output.push_back(headMagic[i]);
	}

	/* Copy the jfif app0 marker to the output buffer */
	for(size_t i = 0; i < 0x10; ++i)
	{
		output.push_back(jfifApp0[i]);
	}
}

/**
 * Write the frame header containing the quantification tables used
 *
 * @param output_buf the output buffer
 * @param w the width of the image
 * @param h the height of the image
 */
void JPEGEncoder::write_frame_header(std::vector<char>& output_buf, size_t w, size_t h)
{
	write_quant_table(output_buf, 0);	/* Y Channel */
	write_quant_table(output_buf, 1);	/* Cb/Cr Channel */
	write_sof(output_buf, w, h);
}

/**
 * Export the quantification table
 *
 * @param output_buf the output buffer
 * @param index the index of the quantification table
 */
void JPEGEncoder::write_quant_table(std::vector<char>& output_buf, int index)
{
	quantification_table_t *qtblptr;
	size_t i;
	qtblptr = &this->m_quant_tbls[index];

	write_marker(output_buf, 0xDB);
	write_2byte(output_buf, 0x40 + 1 + 2);
	write_byte(output_buf, index);
	for(i = 0; i < 0x40; ++i)
	{
		unsigned int qval = qtblptr->value[jpeg_natural_order[i]];
		write_byte(output_buf, (int)(qval & 0xFF));
	}
}


/**
 * Export the huffman tables
 *
 * @param output_buf the output buffer
 * @param index the index of the quantification table
 * @param is_ac flag, true if ac table shall be exported, false if dc
 */
void JPEGEncoder::write_huffman_table(std::vector<char>& output_buf, int index, unsigned char is_ac)
{
	huffman_table_t *htblptr;
	size_t length, i;

	if(is_ac) {
		htblptr = &this->m_ac_huff_tbls[index];
		index += 0x10;
	} else {
		htblptr = &this->m_dc_huff_tbls[index];
	}

	write_marker(output_buf, 0xC4);
	length = 0;
	for(i = 1; i < 0x11; ++i)
		length += htblptr->bits[i];

	/* Write section header containing number of bytes the secion contains */
	write_2byte(output_buf, length + 2 + 1 + 0x10);

	/* output the number of bytes consisting of the index, the bits and the values */
	write_byte(output_buf, index);
	for(i = 1; i < 0x11; ++i)
		write_byte(output_buf, htblptr->bits[i]);
	for(i = 0; i < length; ++i)
		write_byte(output_buf, htblptr->value[i]);
}


/**
 * Write the sos marker
 *
 * @param output_buf the output buffer to use
 */
void JPEGEncoder::write_sos(std::vector<char>& output_buf)
{
	write_marker(output_buf, 0xDA);
	write_2byte(output_buf, 2 * 0x3 + 2 + 1 + 3);
	write_byte(output_buf, 0x3);			/* number of components */

	/* Y Channel */
	write_byte(output_buf, 1);				/* component id */
	write_byte(output_buf, (0 << 0x4) + 0);	/* ( dc_tbl_no << 0x4 ) + ac_tbl_no */

	/* Cb Channel */
	write_byte(output_buf, 2);				/* component id */
	write_byte(output_buf, (1 << 0x4) + 1);	/* ( dc_tbl_no << 0x4 ) + ac_tbl_no */

	/* Cr Channel */
	write_byte(output_buf, 3);				/* component id */
	write_byte(output_buf, (1 << 0x4) + 1);	/* ( dc_tbl_no << 0x4 ) + ac_tbl_no */

	/* End of sos section */
	write_byte(output_buf, 0);
	write_byte(output_buf, 0x3F);
	write_byte(output_buf, 0);
}

/**
 * Write the scan header containing the huffman tables
 *
 * @param output_buf the output buffer to use
 */
void JPEGEncoder::write_scan_header(std::vector<char>& output_buf)
{
	/* Y Channel */
	this->write_huffman_table(output_buf, 0, 0);
	this->write_huffman_table(output_buf, 0, 1);

	/* Cb / Cr Channel */
	this->write_huffman_table(output_buf, 1, 0);
	this->write_huffman_table(output_buf, 1, 1);

	/* Write sos marker */
	this->write_sos(output_buf);
}

/**
 * Write the SOF Part containing the sampling parameters and image size
 *
 * @param output_buf the output buffer to use
 * @param w image width
 * @param h image height
 */
void JPEGEncoder::write_sof(std::vector<char>& output_buf, size_t w, size_t h)
{
	write_marker(output_buf, 0xC0);
	write_2byte(output_buf, 3 * 0x3 + 2 + 5 + 1);
	write_byte(output_buf, 0x8);
	write_2byte(output_buf, h);
	write_2byte(output_buf, w);
	write_byte(output_buf, 0x3);

	/* Y Channel */
	write_byte(output_buf, 0x1);
	write_byte(output_buf, (0x2 << 4) + 0x2);
	write_byte(output_buf, 0);

	/* Cb Channel */
	write_byte(output_buf, 0x2);
	write_byte(output_buf, (0x1 << 4) + 0x1);
	write_byte(output_buf, 1);

	/* Cr Channel */
	write_byte(output_buf, 0x3);
	write_byte(output_buf, (0x1 << 4) + 0x1);
	write_byte(output_buf, 1);
}

/*
 *  NOTE: this algorithm is part of the libjpeg-turbo project
 *  https://github.com/libjpeg-turbo/libjpeg-turbo/
 */
#define EMIT_BYTE() { \
	unsigned char c; \
	put_bits -= 8; \
	c = (unsigned char)(put_buffer >> put_bits); \
	outputbuf.push_back(c); \
	if (c == 0xFF)  /* need to stuff a zero byte? */ \
		outputbuf.push_back((char)0); \
 }
#define CHECKBUF15() { \
	if (put_bits > 0xF) { \
		EMIT_BYTE() \
		EMIT_BYTE() \
	} \
}
#define PUT_BITS(code, size) { \
	put_bits += size; \
	put_buffer = (put_buffer << size) | code; \
}
#define EMIT_BITS(code, size) { \
	PUT_BITS(code, size) \
	CHECKBUF15() \
}
#define EMIT_CODE(code, size) { \
	temp2 &= (((int) 1)<<nbits) - 1; \
	PUT_BITS(code, size) \
	CHECKBUF15() \
	PUT_BITS(temp2, nbits) \
	CHECKBUF15() \
 }

/**
 * Encode the entropy of a single block
 *
 * @param block the block
 * @param table_index the table index for the huffman tables to use
 * @param last_dc_val the last dc value from the previous block
 * @param outputbuf the output buffer to use
 * @param state current bits to be exported from the entropy
 */
void JPEGEncoder::encode_entropy_single_block(short *block, int table_index, int last_dc_val, std::vector<char>& outputbuf, entropy_state_t& state)
{
	int temp, temp2, temp3, r, code, size, put_bits, nbits, code_0xf0, size_0xf0;
	size_t put_buffer;
	derived_huffman_table_t *dcd;
	derived_huffman_table_t *acd;

	/* init values */
	dcd = &this->m_dc_derived_tbls[table_index];
	acd = &this->m_ac_derived_tbls[table_index];
	code_0xf0 = acd->code[0xf0];
	size_0xf0 = acd->length[0xf0];

	put_buffer = state.buffer;
	put_bits = state.bits;


	temp = temp2 = block[0] - last_dc_val;
	temp3 = temp >> (8 * sizeof(int) - 1);
	temp ^= temp3;
	temp -= temp3;

	temp2 += temp3;
	nbits = nbits_table[temp];

	code = dcd->code[nbits];
	size = dcd->length[nbits];
	EMIT_BITS(code, size)

	temp2 &= (((long) 1) << nbits) - 1;
	EMIT_BITS(temp2, nbits)

	/* run length encoding */
	r = 0;

	/* Run length encoding macro */
	#define kloop(k) {  \
		if ((temp = block[k]) == 0) { \
			r++; \
		} else { \
			temp2 = temp; \
			temp3 = temp >> (8 * sizeof(int) - 1); \
			temp ^= temp3; \
			temp -= temp3; \
			temp2 += temp3; \
			nbits = nbits_table[temp]; \
			/* if run length > 15, must emit special run-length-16 codes (0xF0) */ \
			while (r > 15) { \
				EMIT_BITS(code_0xf0, size_0xf0) \
				r -= 16; \
			} \
			/* Emit Huffman symbol for run length / number of bits */ \
			temp3 = (r << 4) + nbits;  \
			code = acd->code[temp3]; \
			size = acd->length[temp3]; \
			EMIT_CODE(code, size) \
			r = 0;  \
		} \
	}

	/* Do run length encoding in zig zag pattern */
	kloop(1);   kloop(8);   kloop(16);  kloop(9);   kloop(2);   kloop(3);
	kloop(10);  kloop(17);  kloop(24);  kloop(32);  kloop(25);  kloop(18);
	kloop(11);  kloop(4);   kloop(5);   kloop(12);  kloop(19);  kloop(26);
	kloop(33);  kloop(40);  kloop(48);  kloop(41);  kloop(34);  kloop(27);
	kloop(20);  kloop(13);  kloop(6);   kloop(7);   kloop(14);  kloop(21);
	kloop(28);  kloop(35);  kloop(42);  kloop(49);  kloop(56);  kloop(57);
	kloop(50);  kloop(43);  kloop(36);  kloop(29);  kloop(22);  kloop(15);
	kloop(23);  kloop(30);  kloop(37);  kloop(44);  kloop(51);  kloop(58);
	kloop(59);  kloop(52);  kloop(45);  kloop(38);  kloop(31);  kloop(39);
	kloop(46);  kloop(53);  kloop(60);  kloop(61);  kloop(54);  kloop(47);
	kloop(55);  kloop(62);  kloop(63);

	if(r > 0) {
		code = acd->code[0];
		size = acd->length[0];
		EMIT_BITS(code, size);
	}

	/* Store the current state back in the global one */
	state.bits = put_bits;
	state.buffer = put_buffer;
}

/**
 * Do a entropy encoding for a super block containing of four luminance blocks and
 * for the cb/cr one chrominance block each
 *
 * @param mcu_buffer pointer to the blocks
 * @param outputbuf the output buffer
 * @param state the entropy state
 */
void JPEGEncoder::encode_entropy(short *mcu_buffer[0x6], std::vector<char>& outputbuf, entropy_state_t& state)
{
	const static unsigned char mcu_membership[0x6] = {0x0, 0x0, 0x0, 0x0, 0x1, 0x2};
	const static unsigned char table_index[0x6] = {0x0, 0x0, 0x0, 0x0, 0x1, 0x1};
	size_t i, ci;

	/* Perform encoding on each block */
	for(i = 0; i < 0x6; ++i)
	{
		ci = mcu_membership[i];
		this->encode_entropy_single_block(mcu_buffer[i], table_index[i], state.last_dc_val[ci], outputbuf, state);
		state.last_dc_val[ci] = mcu_buffer[i][0];
	}
}





// =========================================================================
//
// Setup routine for the encoder
//
// =========================================================================

/**
 * Create the encoder
 *
 * @param quality the quality to use (clamped between 1 and 100)
 */
void JPEGEncoder::create_encoder(unsigned char quality)
{
	this->set_quality_setting(quality);
	this->create_huffman_tables();
	this->create_dct_division_tables();
	this->create_derived_huffman_tables();
}

/**
 * Set the quality in the quantification tables
 *
 * @param quality quality the quality to use (clamped between 1 and 100)
 */
void JPEGEncoder::set_quality_setting(unsigned char quality)
{
	unsigned char q;
	if(quality <= 0)
	{
		q = 1;
	}
	else if(quality > 100)
	{
		q = 100;
	}
	else if(quality < 50)
	{
		q = 5000 / quality;
	}
	else
	{
		q = 200 - (quality << 0x1);
	}

	/* Create quantification tables for luminance and chrominance */
	this->create_quant_table(0, q, std_luminance_quant_tbl);
	this->create_quant_table(1, q, std_chrominance_quant_tbl);
}

/**
 * Create the quantification tables for the encoder based on the given scale factor
 *
 * @param table_id the id of the quantification table in the encoder
 * @param scale the scale to use (quality setting)
 * @param base_table the base table to scale
 */
void JPEGEncoder::create_quant_table(int table_idx, unsigned char scale, const unsigned int *base_table)
{
	quantification_table_t *tblptr;
	size_t i;
	long temp;

	tblptr = &this->m_quant_tbls[table_idx];
	for(i = 0; i < 0x40; ++i)
	{
		temp = ((long)base_table[i] * scale + 50) / 100;
		if(temp <= 0)
		{
			temp = 1;
		}
		else if(temp > 0xFF)
		{
			temp = 0xFF;
		}
		tblptr->value[i] = (unsigned char)temp;
	}
}

/**
 * Create the huffman tables for the given encoder
 */
void JPEGEncoder::create_huffman_tables(void)
{
	/* Luminance huffman table */
	this->add_huffman_table(&this->m_dc_huff_tbls[0], bits_dc_luminance, value_dc_luminance);
	this->add_huffman_table(&this->m_ac_huff_tbls[0], bits_ac_luminance, value_ac_luminance);

	/* Chrominance huffman table */
	this->add_huffman_table(&this->m_dc_huff_tbls[1], bits_dc_chrominance, value_dc_chrominance);
	this->add_huffman_table(&this->m_ac_huff_tbls[1], bits_ac_chrominance, value_ac_chrominance);
}

/**
 * Add the huffman table to the encoder, count the number of bits
 * and copy the number of values accordingly to the huffman table
 *
 * @param tblptr huffman table to use
 * @param bits length
 * @param values the values
 */
void JPEGEncoder::add_huffman_table(huffman_table_t *tblptr, const unsigned char *bits, const unsigned char *values)
{
	size_t len;
	int n = 0;

	/* copy the bits */
	memcpy(tblptr->bits, bits, sizeof(tblptr->bits));

	/* count the length */
	for(len = 0; len < 0x11; ++len)
		n += bits[len];

	/* set table to zero */
	memset(tblptr->value, 0, sizeof(tblptr->value));

	/* copy length many values */
	memcpy(tblptr->value, values, n * sizeof(unsigned char));
}

/**
 * Create the dct divisor tables
 */
void JPEGEncoder::create_dct_division_tables(void)
{
	size_t i;
	quantification_table_t *qtblptr;
	short *dtblptr;

	/* Y Channel */
	qtblptr = &this->m_quant_tbls[0];
	dtblptr = this->m_fdct_divisors[0];
	for(i = 0; i < 0x40; ++i)
	{
		compute_reciprocal(qtblptr->value[i] << 0x3, &dtblptr[i]);
	}

	/* Cb/Cr channel, since they share the table the computation
	 * needs to be done only once */
	qtblptr = &this->m_quant_tbls[1];
	dtblptr = this->m_fdct_divisors[1];
	for(i = 0; i < 0x40; ++i)
	{
		compute_reciprocal(qtblptr->value[i] << 0x3, &dtblptr[i]);
	}
}

/**
 * Create the derived huffman tables
 */
void JPEGEncoder::create_derived_huffman_tables(void)
{
	/* Y Channel */
	this->derive_huffman_table(1, &this->m_dc_huff_tbls[0], &this->m_dc_derived_tbls[0]);
	this->derive_huffman_table(0, &this->m_ac_huff_tbls[0], &this->m_ac_derived_tbls[0]);

	/* Cb/Cr channel, since they share the table the computation
	 * needs to be done only once */
	this->derive_huffman_table(1, &this->m_dc_huff_tbls[1], &this->m_dc_derived_tbls[1]);
	this->derive_huffman_table(0, &this->m_ac_huff_tbls[1], &this->m_ac_derived_tbls[1]);
}

/**
 * Derive the huffman tables
 *
 * @param is_dc flag whether current table is dc or ac
 * @param table_idx table index to use
 * @param pointer to the derived huffman table to be filled
 */
void JPEGEncoder::derive_huffman_table(unsigned char is_dc, huffman_table_t *htblptr, derived_huffman_table_t *dhtblptr)
{
	int p, i, l, lastp, si;
	char huffsize[0x101];
	unsigned int huffcode[0x101];
	unsigned int code;

	/* Figure C.1: make table of Huffman code length for each symbol */
	p = 0;
	for (l = 1; l < 0x11; l++)
	{
		i = (int) htblptr->bits[l];
		while (i--)
		{
			huffsize[p++] = (char) l;
		}
	}
	huffsize[p] = 0;
	lastp = p;

	/* Figure C.2: generate the codes themselves */
	code = 0;
	si = huffsize[0];
	p = 0;
	while (huffsize[p])
	{
		while (((int) huffsize[p]) == si)
		{
			huffcode[p++] = code;
			code++;
		}
		code <<= 1;
		si++;
	}

	/* Figure C.3: generate encoding tables */
	memset(dhtblptr->length, 0, sizeof(dhtblptr->length));

	for (p = 0; p < lastp; ++p)
	{
		i = htblptr->value[p];
		dhtblptr->code[i] = huffcode[p];
		dhtblptr->length[i] = huffsize[p];
	}
}


}	/* end of namespace jpeg */
