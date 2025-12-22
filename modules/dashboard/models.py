from django.db import models


class AnalysisRun(models.Model):
    session_key = models.CharField(max_length=64, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    analysis_type = models.CharField(max_length=64)
    params_json = models.JSONField(default=dict)
    summary_json = models.JSONField(default=dict)

    def __str__(self) -> str:
        return f"{self.session_key} - {self.analysis_type} - {self.created_at.isoformat()}"
