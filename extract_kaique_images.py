#!/usr/bin/env python3
"""Extract images from Kaique thesis PDF"""

import fitz
import os
import glob

# Find the PDF
pdfs = glob.glob("*.pdf")
print(f"Found PDFs: {pdfs}")

for pdf_file in pdfs:
    if "Kaique" in pdf_file or "Disserta" in pdf_file:
        print(f"Opening: {pdf_file}")
        doc = fitz.open(pdf_file)
        os.makedirs('kaique_images', exist_ok=True)

        img_count = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            images = page.get_images()
            for img_idx, img in enumerate(images):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n >= 5:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    fname = f'kaique_images/page{page_num+1}_img{img_idx+1}.png'
                    pix.save(fname)
                    img_count += 1
                    print(f'Saved {fname} ({pix.width}x{pix.height})')
                except Exception as e:
                    print(f'Error on page {page_num+1}: {e}')

        print(f'\nTotal images extracted: {img_count}')
        print(f'Total pages: {len(doc)}')
        break
