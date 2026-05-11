import os
import pandas as pd
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp_uploads")

# Global cache for predictions
_prediction_cache = {}


def set_prediction_cache(file_id: str, predictions: list, model_used: str):
    _prediction_cache[file_id] = {
        "predictions": predictions,
        "model_used": model_used,
        "timestamp": datetime.now().isoformat()
    }


def get_prediction_cache(file_id: str):
    return _prediction_cache.get(file_id, {})


@router.get("/excel/{file_id}")
async def export_to_excel(file_id: str, horizon: int = 10):
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        df = pd.read_csv(file_path)
        cached = get_prediction_cache(file_id)
        predictions = cached.get("predictions", [])
        model_used = cached.get("model_used", "Unknown")

        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Historical Data', index=False)

            if predictions:
                forecast_df = pd.DataFrame({
                    'Step': range(1, min(horizon, len(predictions)) + 1),
                    'Predicted Value': predictions[:horizon],
                    'Model Used': [model_used] * min(horizon, len(predictions))
                })
            else:
                forecast_df = pd.DataFrame({
                    'Step': [1],
                    'Predicted Value': ['Run forecast first'],
                    'Model Used': ['']
                })

            forecast_df.to_excel(writer, sheet_name='Forecast', index=False)

            summary_df = pd.DataFrame({
                'Info': ['Total Rows', 'Columns', 'Horizon', 'Model', 'Generated'],
                'Value': [len(df), len(df.columns), horizon, model_used, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=forecast_{file_id}.xlsx"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel export failed: {str(e)}")


@router.get("/pdf/{file_id}")
async def export_to_pdf(file_id: str, horizon: int = 10):
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        df = pd.read_csv(file_path)
        cached = get_prediction_cache(file_id)
        predictions = cached.get("predictions", [])
        model_used = cached.get("model_used", "Unknown")

        output = io.BytesIO()

        # Create PDF
        doc = SimpleDocTemplate(output, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#667eea'),
            alignment=1,
            spaceAfter=20
        )
        story.append(Paragraph("AutoML Time Series Studio - Forecast Report", title_style))
        story.append(Spacer(1, 10))

        # Report info
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>File ID:</b> {file_id}", styles['Normal']))
        story.append(Paragraph(f"<b>Forecast Horizon:</b> {horizon} steps", styles['Normal']))

        if predictions:
            story.append(Paragraph(f"<b>Model Used:</b> {model_used}", styles['Normal']))

        story.append(Spacer(1, 20))

        # Data summary
        story.append(Paragraph("Data Summary", styles['Heading2']))
        story.append(Paragraph(f"• Total rows: {len(df)}", styles['Normal']))
        story.append(Paragraph(f"• Columns: {', '.join(df.columns.tolist())}", styles['Normal']))
        story.append(Spacer(1, 10))

        # First 10 rows table
        story.append(Paragraph("Sample Data (First 10 rows):", styles['Heading3']))

        # Prepare table data
        table_data = [df.columns.tolist()]
        for i in range(min(10, len(df))):
            row = []
            for col in df.columns:
                val = df.iloc[i][col]
                if isinstance(val, float):
                    row.append(f"{val:.4f}")
                else:
                    row.append(str(val))
            table_data.append(row)

        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Forecast results
        if predictions:
            story.append(Paragraph("Forecast Results", styles['Heading2']))

            forecast_data = [['Step', 'Predicted Value']]
            for i, val in enumerate(predictions[:horizon]):
                forecast_data.append([str(i + 1), f"{val:.4f}"])

            forecast_table = Table(forecast_data)
            forecast_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff6b6b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(forecast_table)
        else:
            story.append(Paragraph("No forecast available. Please run forecast first.", styles['Normal']))

        # Build PDF
        doc.build(story)
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=forecast_report_{file_id}.pdf"}
        )

    except Exception as e:
        import traceback
        error_detail = f"PDF export failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)