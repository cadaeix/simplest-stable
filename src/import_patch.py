

def patch_import_errors():
    import PIL.Image
    if not hasattr(PIL.Image, 'Resampling'):
        PIL.Image.Resampling = PIL.Image

    import PIL._util
    if not hasattr(PIL._util, 'is_path'):
        from pathlib import Path

        def is_path(f):
            return isinstance(f, (bytes, str, Path))
        PIL._util.is_path = is_path

    import PIL.TiffTags
    if not hasattr(PIL.TiffTags, 'IFD'):
        PIL.TiffTags.IFD = 13
    if not hasattr(PIL.TiffTags, 'LONG8'):
        PIL.TiffTags.LONG8 = 16

    import PIL.ImageFile
    if not hasattr(PIL.ImageFile, 'PyEncoder'):
        from PIL.ImageFile import PyCodecState

        class PyCodec:
            def __init__(self, mode, *args):
                self.im = None
                self.state = PyCodecState()
                self.fd = None
                self.mode = mode
                self.init(args)

            def init(self, args):
                """
                Override to perform codec specific initialization

                :param args: Array of args items from the tile entry
                :returns: None
                """
                self.args = args

            def cleanup(self):
                """
                Override to perform codec specific cleanup

                :returns: None
                """
                pass

            def setfd(self, fd):
                """
                Called from ImageFile to set the Python file-like object

                :param fd: A Python file-like object
                :returns: None
                """
                self.fd = fd

            def setimage(self, im, extents=None):
                """
                Called from ImageFile to set the core output image for the codec

                :param im: A core image object
                :param extents: a 4 tuple of (x0, y0, x1, y1) defining the rectangle
                    for this tile
                :returns: None
                """

                # following c code
                self.im = im

                if extents:
                    (x0, y0, x1, y1) = extents
                else:
                    (x0, y0, x1, y1) = (0, 0, 0, 0)

                if x0 == 0 and x1 == 0:
                    self.state.xsize, self.state.ysize = self.im.size
                else:
                    self.state.xoff = x0
                    self.state.yoff = y0
                    self.state.xsize = x1 - x0
                    self.state.ysize = y1 - y0

                if self.state.xsize <= 0 or self.state.ysize <= 0:
                    msg = "Size cannot be negative"
                    raise ValueError(msg)

                if (
                    self.state.xsize + self.state.xoff > self.im.size[0]
                    or self.state.ysize + self.state.yoff > self.im.size[1]
                ):
                    msg = "Tile cannot extend outside image"
                    raise ValueError(msg)
        PIL.ImageFile.PyCodec = PyCodec

        class PyEncoder(PyCodec):
            """
            Python implementation of a format encoder. Override this class and
            add the decoding logic in the :meth:`encode` method.

            See :ref:`Writing Your Own File Codec in Python<file-codecs-py>`
            """

            _pushes_fd = False

            @property
            def pushes_fd(self):
                return self._pushes_fd

            def encode(self, bufsize):
                """
                Override to perform the encoding process.

                :param bufsize: Buffer size.
                :returns: A tuple of ``(bytes encoded, errcode, bytes)``.
                    If finished with encoding return 1 for the error code.
                    Err codes are from :data:`.ImageFile.ERRORS`.
                """
                raise NotImplementedError()

            def encode_to_pyfd(self):
                """
                If ``pushes_fd`` is ``True``, then this method will be used,
                and ``encode()`` will only be called once.

                :returns: A tuple of ``(bytes consumed, errcode)``.
                    Err codes are from :data:`.ImageFile.ERRORS`.
                """
                if not self.pushes_fd:
                    return 0, -8  # bad configuration
                bytes_consumed, errcode, data = self.encode(0)
                if data:
                    self.fd.write(data)
                return bytes_consumed, errcode

            def encode_to_file(self, fh, bufsize):
                """
                :param fh: File handle.
                :param bufsize: Buffer size.

                :returns: If finished successfully, return 0.
                    Otherwise, return an error code. Err codes are from
                    :data:`.ImageFile.ERRORS`.
                """
                errcode = 0
                while errcode == 0:
                    status, errcode, buf = self.encode(bufsize)
                    if status > 0:
                        fh.write(buf[status:])
                return errcode
        PIL.ImageFile.PyEncoder = PyEncoder
