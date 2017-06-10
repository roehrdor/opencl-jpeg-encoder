#ifndef _JPEG_ENCODER_
#define _JPEG_ENCODER_

#include <CL/cl.hpp>
#include <string>
#include <fstream>
#include <streambuf>
#include <vector>
#include <cstdio>
#include "tables.h"

namespace jpeg
{

struct derived_huffman_table
{
	unsigned int code[0x100];
	unsigned char length[0x100];
};
typedef derived_huffman_table derived_huffman_table_t;

struct huffman_table
{
	unsigned char bits[0x11];
	unsigned char value[0x100];
};
typedef struct huffman_table huffman_table_t;

struct entropy_state
{
	size_t buffer;
	int bits;
	int last_dc_val[0x3];
};
typedef struct entropy_state entropy_state_t;

struct quantification_table
{
	unsigned char value[0x40];
};
typedef struct quantification_table quantification_table_t;



class JPEGEncoder
{
private:
	/* Quantification Table, one for luminance, one for chrominance */
	quantification_table_t m_quant_tbls[0x2];

	/* Division Lookup table for DCT, one for luminance, one for chrominance */
	short m_fdct_divisors[0x2][0x100];

	/* entropy encoding */
	derived_huffman_table_t m_dc_derived_tbls[0x2];
	derived_huffman_table_t m_ac_derived_tbls[0x2];
	huffman_table_t m_dc_huff_tbls[0x2];
	huffman_table_t m_ac_huff_tbls[0x2];

	/* OpenCL Context used in this class */
	cl::Context m_context;

	/* OpenCL device to use */
	cl::Device m_device;

	/* OpenCL command queue to use */
	cl::CommandQueue m_queue;

	/* Program containing the kernels */
	cl::Program m_program;

	/* Kernels */
	cl::Kernel m_transformation_kernel;
	cl::Kernel m_downsample_full_kernel;
	cl::Kernel m_downsample_2v2_kernel;
	cl::Kernel m_dct_quant;
	cl::Kernel m_zero_out_right;
	cl::Kernel m_zero_out_bottom;

	/*
	   Look up tables
	   	with size of 3 * 3 * 256

	   	the partial tables are in the following ranges
	   	   0 <= x <  768: red
		 768 <= x < 1536: green
		1536 <= x < 2304: blue

	 	each of the following tables consists of 256 elements,
	 	where each element is a structure as the follows
			struct {
				unsigned int y;
				unsigned int cr;
				unsigned int cb;
			};

	   The values are stored as integers since they are shifted by 16
	   to not lose precision. After summing up all the elements for
	   a channel the result needs to be right-shifted again
	   and then fits into a single unsigned char (1 Byte) value
	*/
	cl::Buffer md_color_conversion_table;

	/* Divisor table for the quantification */
	cl::Buffer md_fdct_divisors;
	cl::Buffer md_fdct_multiplier;
	cl::Buffer md_fdct_sign;
	cl::Buffer md_fdct_indices;
	cl::Buffer md_fdct_descaler;
	cl::Buffer md_fdct_descaler_offset;


	/**
	 * Encode the given image
	 *
	 * @param image pointer to the image data in flat row major layout
	 * @param width of the image
	 * @param height of the image
	 * @param file the output file to store the image at
	 * @param cpu 0 iff cpu shall not be used to compare and validate
	 * @return 0 on success
	 */
	int encode_image(unsigned char* image, size_t width, size_t height, const char * const file, int cpu);

	/**
	 * Prepare the device that runs the encoding process by uploading
	 * the color conversion table and preparing dct, huffman, ...
	 */
	void prepare_device(void);


	/**
	 * Create the encoder
	 *
	 * @param quality the quality to use (clamped between 1 and 100)
	 */
	void create_encoder(unsigned char);

	/**
	 * Set the quality in the quantification tables
	 *
	 * @param quality quality the quality to use (clamped between 1 and 100)
	 */
	void set_quality_setting(unsigned char);

	/**
	 * Create the huffman tables for the given encoder
	 */
	void create_huffman_tables(void);

	/**
	 * Create the dct divisor tables
	 */
	void create_dct_division_tables(void);

	/**
	 * Create the derived huffman tables
	 */
	void create_derived_huffman_tables(void);

	/**
	 * Create the quantification tables for the encoder based on the given scale factor
	 *
	 * @param table_id the id of the quantification table in the encoder
	 * @param scale the scale to use (quality setting)
	 * @param base_table the base table to scale
	 */
	void create_quant_table(int, unsigned char, const unsigned int *);

	/**
	 * Add the huffman table to the encoder, count the number of bits
	 * and copy the number of values accordingly to the huffman table
	 *
	 * @param tblptr huffman table to use
	 * @param bits length
	 * @param values the values
	 */
	void add_huffman_table(huffman_table_t *, const unsigned char *, const unsigned char *);

	/**
	 * Derive the huffman tables
	 *
	 * @param is_dc flag whether current table is dc or ac
	 * @param table_idx table index to use
	 * @param pointer to the derived huffman table to be filled
	 */
	void derive_huffman_table(unsigned char, huffman_table_t *, derived_huffman_table_t *);

	/**
	 * Encode the entropy of a single block
	 *
	 * @param block the block
	 * @param table_index the table index for the huffman tables to use
	 * @param last_dc_val the last dc value from the previous block
	 * @param outputbuf the output buffer to use
	 * @param state current bits to be exported from the entropy
	 */
	void encode_entropy_single_block(short *, int, int, std::vector<char>&, entropy_state_t&);

	/**
	 * Do a entropy encoding for a super block containing of four luminance blocks and
	 * for the cb/cr one chrominance block each
	 *
	 * @param mcu_buffer pointer to the blocks
	 * @param outputbuf the output buffer
	 * @param state the entropy state
	 */
	void encode_entropy(short *mcu_buffer[0x6], std::vector<char>&, entropy_state_t&);

	/**
	 * Write the file header
	 *
	 * @param output_buf the output buffer to use
	 */
	void write_file_header(std::vector<char>&);

	/**
	 * Write the frame header containing the quantification tables used
	 *
	 * @param output_buf the output buffer
	 * @param w the width of the image
	 * @param h the height of the image
	 */
	void write_frame_header(std::vector<char>& output_buf, size_t w, size_t h);

	/**
	 * Export the quantification table
	 *
	 * @param output_buf the output buffer
	 * @param index the index of the quantification table
	 */
	void write_quant_table(std::vector<char>& output_buf, int index);

	/**
	 * Export the huffman tables
	 *
	 * @param output_buf the output buffer
	 * @param index the index of the quantification table
	 * @param is_ac flag, true if ac table shall be exported, false if dc
	 */
	void write_huffman_table(std::vector<char>& output_buf, int index, unsigned char is_ac);

	/**
	 * Write the sos marker
	 *
	 * @param output_buf the output buffer to use
	 */
	void write_sos(std::vector<char>& output_buf);

	/**
	 * Write the scan header containing the huffman tables
	 *
	 * @param output_buf the output buffer to use
	 */
	void write_scan_header(std::vector<char>& output_buf);

	/**
	 * Write the SOF Part containing the sampling parameters and image size
	 *
	 * @param output_buf the output buffer to use
	 * @param w image width
	 * @param h image height
	 */
	void write_sof(std::vector<char>& outputbuf, size_t w, size_t h);

public:

	/**
	 * Create a new encoder
	 *
	 * @param type the device type to use
	 * @param quality the quality setting to use (clamped between 1 and 100)
	 */
	JPEGEncoder(cl_device_type type, unsigned char quality);

	/**
	 * Encode the given image
	 *
	 * @param image pointer to the image data in flat row major layout
	 * @param width of the image
	 * @param height of the image
	 * @param file the output file to store the image at
	 * @return 0 on success
	 */
	int encode_image(unsigned char* image, size_t width, size_t height, const char * const file);
};
}

#endif
