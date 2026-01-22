# version:1.1.2312.9221
import gxipy as gx
import time
from PIL import Image
from ctypes import *
from gxipy.gxidef import *
import numpy
from gxipy.ImageFormatConvert import *
from gxipy.ImageProc import *

def get_best_valid_bits(pixel_format):
    valid_bits = DxValidBit.BIT0_7
    if pixel_format in (GxPixelFormatEntry.MONO8,
                        GxPixelFormatEntry.BAYER_GR8, GxPixelFormatEntry.BAYER_RG8,
                        GxPixelFormatEntry.BAYER_GB8, GxPixelFormatEntry.BAYER_BG8,
                        GxPixelFormatEntry.RGB8, GxPixelFormatEntry.BGR8,
                        GxPixelFormatEntry.R8, GxPixelFormatEntry.B8, GxPixelFormatEntry.G8):
        valid_bits = DxValidBit.BIT0_7
    elif pixel_format in (GxPixelFormatEntry.MONO10, GxPixelFormatEntry.MONO10_PACKED, GxPixelFormatEntry.MONO10_P,
                          GxPixelFormatEntry.BAYER_GR10, GxPixelFormatEntry.BAYER_RG10,
                          GxPixelFormatEntry.BAYER_GB10, GxPixelFormatEntry.BAYER_BG10,
                          GxPixelFormatEntry.BAYER_GR10_P, GxPixelFormatEntry.BAYER_RG10_P,
                          GxPixelFormatEntry.BAYER_GB10_P, GxPixelFormatEntry.BAYER_BG10_P,
                          GxPixelFormatEntry.BAYER_GR10_PACKED, GxPixelFormatEntry.BAYER_RG10_PACKED,
                          GxPixelFormatEntry.BAYER_GB10_PACKED, GxPixelFormatEntry.BAYER_BG10_PACKED):
        valid_bits = DxValidBit.BIT2_9
    elif pixel_format in (GxPixelFormatEntry.MONO12, GxPixelFormatEntry.MONO12_PACKED, GxPixelFormatEntry.MONO12_P,
                          GxPixelFormatEntry.BAYER_GR12, GxPixelFormatEntry.BAYER_RG12,
                          GxPixelFormatEntry.BAYER_GB12, GxPixelFormatEntry.BAYER_BG12,
                          GxPixelFormatEntry.BAYER_GR12_P, GxPixelFormatEntry.BAYER_RG12_P,
                          GxPixelFormatEntry.BAYER_GB12_P, GxPixelFormatEntry.BAYER_BG12_P,
                          GxPixelFormatEntry.BAYER_GR12_PACKED, GxPixelFormatEntry.BAYER_RG12_PACKED,
                          GxPixelFormatEntry.BAYER_GB12_PACKED, GxPixelFormatEntry.BAYER_BG12_PACKED):
        valid_bits = DxValidBit.BIT4_11
    elif pixel_format in (GxPixelFormatEntry.MONO14, GxPixelFormatEntry.MONO14_P,
                          GxPixelFormatEntry.BAYER_GR14, GxPixelFormatEntry.BAYER_RG14,
                          GxPixelFormatEntry.BAYER_GB14, GxPixelFormatEntry.BAYER_BG14,
                          GxPixelFormatEntry.BAYER_GR14_P, GxPixelFormatEntry.BAYER_RG14_P,
                          GxPixelFormatEntry.BAYER_GB14_P, GxPixelFormatEntry.BAYER_BG14_P,
                          ):
        valid_bits = DxValidBit.BIT6_13
    elif pixel_format in (GxPixelFormatEntry.MONO16,
                          GxPixelFormatEntry.BAYER_GR16, GxPixelFormatEntry.BAYER_RG16,
                          GxPixelFormatEntry.BAYER_GB16, GxPixelFormatEntry.BAYER_BG16):
        valid_bits = DxValidBit.BIT8_15
    return valid_bits


def convert_to_special_pixel_format(image_convert, raw_image, pixel_format):
    image_convert.set_dest_format(pixel_format)
    valid_bits = get_best_valid_bits(raw_image.get_pixel_format())
    image_convert.set_valid_bits(valid_bits)

    # create out put image buffer
    buffer_out_size = image_convert.get_buffer_size_for_conversion(raw_image)
    output_image_array = (c_ubyte * buffer_out_size)()
    output_image = addressof(output_image_array)

    # convert to pixel_format
    image_convert.convert(raw_image, output_image, buffer_out_size, False)
    if output_image is None:
        print('Pixel format conversion failed')
        return

    return output_image_array, buffer_out_size

def capture(device, image_convert):
    raw_image = device.data_stream[0].get_image()  # 单位是ms
    if raw_image is None:
        print("Getting image failed.")
        return

    if raw_image.get_pixel_format() not in (
            GxPixelFormatEntry.MONO8, GxPixelFormatEntry.R8, GxPixelFormatEntry.B8, GxPixelFormatEntry.G8):
        mono_image_array, mono_image_buffer_length = convert_to_special_pixel_format(image_convert,
                                                                                     GxPixelFormatEntry.MONO8)
        if mono_image_array is None:
            return
        # create numpy array with data from rgb image
        numpy_image = numpy.frombuffer(mono_image_array, dtype=numpy.ubyte, count=mono_image_buffer_length). \
            reshape(raw_image.frame_data.height, raw_image.frame_data.width)
    else:
        numpy_image = raw_image.get_numpy_array()

    if numpy_image is None:
        return
    return numpy_image
