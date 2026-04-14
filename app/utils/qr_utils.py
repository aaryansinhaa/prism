"""QR code generation utilities for shareable URLs."""

from __future__ import annotations

import base64
import io

try:
    import qrcode
    from qrcode.image.svg import SvgPathImage

    _QR_AVAILABLE = True
except ImportError:
    _QR_AVAILABLE = False


def generate_qr_data_uri(url: str, box_size: int = 10) -> str:
    """Generate a QR code as a data URI SVG string for inline embedding.

    Args:
        url: The URL to encode in the QR code.
        box_size: Size of each box in pixels.

    Returns:
        Data URI string suitable for direct HTML <img src="..."> embedding.
    """
    if not _QR_AVAILABLE:
        raise RuntimeError("qrcode library not available")

    try:
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=box_size,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)

        # Generate as SVG for pure text output
        img = qr.make_image(image_factory=SvgPathImage)

        # Convert SVG to data URI
        buffer = io.BytesIO()
        img.save(buffer)
        buffer.seek(0)
        svg_content = buffer.getvalue().decode("utf-8")

        svg_base64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{svg_base64}"
    except Exception as exc:
        raise RuntimeError(f"Failed to generate QR code: {exc}") from exc
