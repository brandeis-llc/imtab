from typing import Tuple, List
from os.path import join as pjoin
import numpy as np
import os
import attr
import pytesseract
import fitz
import cv2


@attr.s(auto_attribs=True)
class TableExtractor(object):
    images: List[np.ndarray] = attr.ib()
    coordinates_lst: List[List[Tuple]] = attr.ib()

    @classmethod
    def from_image_file(cls, image_file: str, rm_border: bool) -> "TableExtractor":
        # initialize from a single image file
        image = cv2.imread(image_file)
        raw_image = image.copy()
        if rm_border:
            image = cls.remove_border(image)
        _, coordinates = cls.image2bbox(image)
        return cls([raw_image], [coordinates])

    @classmethod
    def from_image_folder(cls, image_folder: str, rm_border: bool) -> "TableExtractor":
        # initialize from a folder of images
        images = []
        coordinates = []
        for root, _, files in os.walk(image_folder):
            for num, file in enumerate(sorted(files)):
                if file.endswith("png"):
                    image = cv2.imread(pjoin(root, file))
                    images.append(image)
                    if rm_border:
                        image = cls.remove_border(image)
                    _, co = cls.image2bbox(image)
                    coordinates.append(co)
        return cls(images, coordinates)

    @classmethod
    def from_image_bytes(
        cls, bytes_lst: List[bytes], rm_border: bool
    ) -> "TableExtractor":
        # initialize from bytes of images (output from fitz)
        images = []
        coordinates = []
        for img_bytes in bytes_lst:
            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
            images.append(image)
            if rm_border:
                image = cls.remove_border(image)
            _, co = cls.image2bbox(image)
            coordinates.append(co)
        return cls(images, coordinates)

    @staticmethod
    def _rm_empty_and_flatten_text(boxes_text):
        non_empty = []
        for texts in boxes_text:
            for i, text in enumerate(texts):
                if text:
                    non_empty.append(text)
        return non_empty

    def ocr(self):
        # apply OCR on each bbox according to the coordinates
        boxes_text = []
        for i, (image, coordinates) in enumerate(
            zip(self.images, self.coordinates_lst)
        ):
            texts = []
            for cnt in coordinates:
                x, y, w, h = cnt
                cropped = image[y : y + h, x : x + w]
                text = " ".join(pytesseract.image_to_string(cropped).strip().split())
                texts.append(text)
            boxes_text.append(texts)
        return tuple(boxes_text)

    def box2file(self, out_file_prefix: str):
        # draw bounding boxes on the images and output
        for i, (image, cnts) in enumerate(zip(self.images, self.coordinates_lst)):
            for cnt in cnts:
                x, y, w, h = cnt
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(out_file_prefix.rsplit(".", 1)[0] + str(i) + ".png", image)

    @staticmethod
    def image2bbox(image: "np.ndarray") -> Tuple["np.ndarray", List[Tuple]]:
        """
        get coordinates of bounding boxes on an image and a copy of the raw image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        kernel = (1100, 5)  # configurable
        ret, thresh1 = cv2.threshold(
            gray, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV
        )
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        contours, hierarchy = cv2.findContours(
            dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        coordinates = [cv2.boundingRect(cnt) for cnt in contours]
        sorted_coordinates = sorted(coordinates, key=lambda k: k[1])
        return image, sorted_coordinates

    @staticmethod
    def remove_border(image: "np.ndarray") -> "np.ndarray":
        """
        remove border lines from the image
        :param image:
        :return:
        """
        bi_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bi_image = cv2.threshold(
            bi_image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
        horizontal_img = bi_image.copy()
        vertical_img = bi_image.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))  # configurable
        horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
        horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))  # configurable
        vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
        vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

        mask_img = horizontal_img + vertical_img
        no_border = np.bitwise_or(255 - bi_image, mask_img)
        return no_border


def main():
    pdf_file = "data/NewsHour_Rundown_Samples_1.pdf"
    doc = fitz.Document(pdf_file)
    page = doc.loadPage(0)  # select the page
    page_bytes = page.getPixmap(matrix=fitz.Matrix(3, 3)).getImageData()
    te = TableExtractor.from_image_bytes(
        [page_bytes], rm_border=True
    )  # init from bytes
    te.box2file("processed")  # output to image file
    text = te.ocr()  # apply ocr on each bboxes


if __name__ == "__main__":
    main()
