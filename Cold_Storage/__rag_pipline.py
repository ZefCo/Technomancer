def _load_pdf_multimodal(file_path: pathlib.Path | str, vision: str, fallback_to_pdfplumber: bool = True) -> list:
    '''
    Loads a PDF with the following strategy:
    - Scores each page of the PDF
    - Pages with good scores go to the normal route of scoring
    - Pages with bad scores get turned into PNGs and their text extracted
        - These pages are then examined by a Vision LLM that interprets their content
    '''
    documents = []

    # if isinstance(file_path, str): file_path = pathlib.Path(file_path)

    with open(file_path, "rb") as file:
        with pdfplumber.open(file) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Loading {str(file_path)} | {total_pages} pages | Vision Model: {vision}")

            page: PageClass
            for i, page in enumerate(pdf.pages):
                chars = page.chars
                text_standard = page.extract_text() or ""

                angled_ratio = 0.0
                if chars:
                    angled = sum(1 for c in chars if abs(c.get('matrix', (0,0,0,0,0,0))[1]) >= 0.1)
                    angled_ratio = angled / len(chars)

                words = text_standard.split()
                avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0
                is_suspect = (
                    angled_ratio > 0.2
                    or avg_word_len < 2.5
                    or avg_word_len > 12
                    or (len(text_standard) < 100 and len(page.images) > 0)
                )

                if is_suspect or not fallback_to_pdfplumber:
                    logger.info(f"Page {i}: using vision extraction | angled: {angled_ratio} | average word length: {avg_word_len}")

                    try:
                        text = _extract_page_multimodal(page, vision)
                        extraction_method = "multimodal"
                    except Exception as e:
                        logger.error(f"Vision extraction failed on page {i} | {type(e)} | {e} | {str(file_path)}")
                        text = text_standard
                        extraction_method = "pdfplumber fallback"
                
                else:
                    text = text_standard
                    extraction_method = "pdfplumber"

                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata = {
                            "source": str(file_path),
                            "page": i,
                            "extraction_method": extraction_method,
                            "angled_ratio": round(angled_ratio, 3),
                            "quality_pass": not is_suspect,
                        }
                    ))

                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{total_pages} pages processed")

        vision_count = sum(1 for d in documents if d.metadata.get("extraction_method") == "multimodal")

        logger.info(f"Extraction complete | PDFPlumber: {len(documents) - vision_count} | Vision: {vision_count}")

    return documents


def _load_document_column_aware_alt(file_path: pathlib.Path) -> list:
    '''
    '''
    # import pdfplumber
    documents = []
    with pdfplumber.open(str(file_path)) as pdf:
        for page in pdf.pages:
            width, height = page.width, page.height

            left_bbox = (0, 0, width / 2, height)
            right_bbox = (width / 2, 0, width, height)

            left_col = page.within_bbox(left_bbox).extract_text()
            right_col = page.within_bbox(right_bbox).extract_text()

            if left_col: documents.append(left_col)
            if right_col: documents.append(right_col)

    return documents