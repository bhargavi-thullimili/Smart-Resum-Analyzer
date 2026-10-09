import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from docx import Document
from pypdf import PdfWriter

import app as app_module


class ResumeAnalyzerTests(unittest.TestCase):
    def setUp(self):
        app_module.app.testing = True
        self.client = app_module.app.test_client()

    @staticmethod
    def make_docx(contents):
        output = BytesIO()
        document = Document()
        document.add_paragraph(contents)
        document.save(output)
        return output.getvalue()

    @staticmethod
    def make_pdf(contents):
        content_stream = f"BT /F1 12 Tf 72 720 Td ({contents}) Tj ET".encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(content_stream)).encode("ascii") + b" >>\nstream\n" + content_stream + b"\nendstream",
        ]
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for object_number, body in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{object_number} 0 obj\n".encode("ascii") + body + b"\nendobj\n")

        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        pdf.extend(
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
        )
        return bytes(pdf)

    @staticmethod
    def mock_groq_response(groq_factory):
        result = {
            "score": 50,
            "summary": "The resume demonstrates Python and Flask experience.",
            "matched_keywords": ["Python", "Flask"],
            "missing_keywords": ["SQL", "React"],
            "recommendations": ["Add accurate SQL project details if applicable."],
        }
        message = SimpleNamespace(content=json.dumps(result))
        choice = SimpleNamespace(message=message)
        completion = SimpleNamespace(choices=[choice])
        groq_factory.return_value.chat.completions.create.return_value = completion

    def test_home_renders_upload_form(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Job description", response.data)
        self.assertIn(b"sent to Groq", response.data)
        self.assertIn(b"groq_consent", response.data)

    def test_docx_is_analyzed_without_being_saved(self):
        upload_dir = Path(app_module.app.root_path) / "uploads"
        before = set(upload_dir.rglob("*")) if upload_dir.exists() else set()
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}), patch("app.Groq") as groq_factory:
            self.mock_groq_response(groq_factory)
            response = self.client.post(
                "/",
                data={
                    "resume": (BytesIO(self.make_docx("Built APIs with Python and Flask.")), "resume.docx"),
                    "job_description": "Python, Flask, SQL, React",
                    "groq_consent": "yes",
                },
                content_type="multipart/form-data",
            )
            call = groq_factory.return_value.chat.completions.create.call_args
        after = set(upload_dir.rglob("*")) if upload_dir.exists() else set()

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"50%", response.data)
        self.assertIn(b"Python", response.data)
        self.assertIn(b"SQL", response.data)
        self.assertEqual(call.kwargs["model"], app_module.GROQ_MODEL)
        self.assertIn(b'id="download-report"', response.data)
        self.assertIn(b"Add accurate SQL project details", response.data)
        self.assertEqual(before, after)

    def test_pdf_is_analyzed(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}), patch("app.Groq") as groq_factory:
            self.mock_groq_response(groq_factory)
            response = self.client.post(
                "/",
                data={
                    "resume": (BytesIO(self.make_pdf("Python Flask")), "resume.pdf"),
                    "job_description": "Python, Flask, SQL, React",
                    "groq_consent": "yes",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"50%", response.data)
        self.assertIn(b"Flask", response.data)

    def test_legacy_doc_is_rejected(self):
        response = self.client.post(
            "/",
            data={
                "resume": (BytesIO(b"legacy document"), "resume.doc"),
                "job_description": "Python experience",
                "groq_consent": "yes",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Upload a PDF or DOCX file.", response.data)

    def test_job_description_is_required(self):
        response = self.client.post(
            "/",
            data={
                "resume": (BytesIO(self.make_docx("Python")), "resume.docx"),
                "job_description": "",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Add the job description", response.data)

    def test_missing_groq_key_returns_setup_message(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}), patch("app.Groq") as groq_factory:
            response = self.client.post(
                "/",
                data={
                    "resume": (BytesIO(self.make_docx("Python")), "resume.docx"),
                    "job_description": "Python experience",
                    "groq_consent": "yes",
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn(b"GROQ_API_KEY", response.data)
        groq_factory.assert_not_called()

    def test_consent_is_required(self):
        response = self.client.post(
            "/",
            data={
                "resume": (BytesIO(self.make_docx("Python")), "resume.docx"),
                "job_description": "Python experience",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Confirm that your resume", response.data)

    def test_pdf_without_selectable_text_is_rejected(self):
        output = BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        writer.write(output)
        response = self.client.post(
            "/",
            data={
                "resume": (BytesIO(output.getvalue()), "scanned.pdf"),
                "job_description": "Python experience",
                "groq_consent": "yes",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No selectable text found", response.data)


if __name__ == "__main__":
    unittest.main()
