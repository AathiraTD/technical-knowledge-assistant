"""OpenTelemetry export — lightweight, no SDK, fire-and-forget."""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from assistant import otel_export


class TestOtelExport(unittest.TestCase):
    """Test OTLP export functions."""

    def test_export_trace_builds_otel_format(self):
        """Trace export builds valid OTel JSON."""
        # Mock the send function to capture the payload
        with patch.object(otel_export, '_send_otlp') as mock_send:
            with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317'}):
                otel_export.export_trace(
                    trace_id='abc123', span_id='xyz789', parent_span_id='parent123',
                    name='answer', started_at='2026-09-16T14:30:00Z', duration_ms=5000,
                    status='ok', attributes={'model': 'qwen3.5:4b', 'count': 42}
                )

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            self.assertEqual(call_args[0][0], 'traces')
            payload = call_args[0][1]

            # Verify structure
            self.assertIn('resourceSpans', payload)
            self.assertEqual(len(payload['resourceSpans']), 1)
            span_data = payload['resourceSpans'][0]['scopeSpans'][0]['spans'][0]
            self.assertEqual(span_data['name'], 'answer')
            self.assertEqual(span_data['traceId'], 'abc123')
            self.assertEqual(span_data['spanId'], 'xyz789')

    def test_export_trace_not_called_if_endpoint_not_set(self):
        """No export if OTEL_EXPORTER_OTLP_ENDPOINT not set."""
        with patch.object(otel_export, '_send_otlp') as mock_send:
            with patch.dict(os.environ, {}, clear=True):
                otel_export.export_trace(
                    trace_id='abc', span_id='xyz', parent_span_id='',
                    name='test', started_at='2026-09-16T14:30:00Z', duration_ms=100,
                    status='ok', attributes={}
                )

            mock_send.assert_not_called()

    def test_export_metrics_builds_otel_format(self):
        """Metrics export builds valid OTel JSON."""
        with patch.object(otel_export, '_send_otlp') as mock_send:
            with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317'}):
                metrics_data = {
                    'metrics': [
                        {
                            'name': 'test_metric',
                            'gauge': {'dataPoints': [
                                {'attributes': [], 'asInt': 42}
                            ]}
                        }
                    ]
                }
                otel_export.export_metrics(metrics_data)

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            self.assertEqual(call_args[0][0], 'metrics')
            payload = call_args[0][1]
            self.assertIn('resourceMetrics', payload)

    def test_export_log_builds_otel_format(self):
        """Log export builds valid OTel JSON."""
        with patch.object(otel_export, '_send_otlp') as mock_send:
            with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317'}):
                otel_export.export_log(
                    level='INFO', message='Test message', timestamp='2026-09-16T14:30:00Z',
                    trace_id='abc123', span_id='xyz789', attributes={'key': 'value'}
                )

            mock_send.assert_called_once()
            call_args = mock_send.call_args
            self.assertEqual(call_args[0][0], 'logs')
            payload = call_args[0][1]
            self.assertIn('resourceLogs', payload)
            log_record = payload['resourceLogs'][0]['scopeLogs'][0]['logRecords'][0]
            self.assertEqual(log_record['severityText'], 'INFO')
            self.assertEqual(log_record['traceId'], 'abc123')

    def test_resource_includes_service_metadata(self):
        """Resource metadata is included in export."""
        with patch.object(otel_export, '_send_otlp') as mock_send:
            with patch.dict(os.environ, {
                'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317',
                'OTEL_SERVICE_NAME': 'test-service',
                'OTEL_SERVICE_VERSION': '2.0.0'
            }):
                otel_export.export_trace(
                    trace_id='a', span_id='b', parent_span_id='',
                    name='test', started_at='2026-09-16T14:30:00Z', duration_ms=1,
                    status='ok', attributes={}
                )

            payload = mock_send.call_args[0][1]
            resource = payload['resourceSpans'][0]['resource']
            attrs = {a['key']: a['value']['stringValue'] for a in resource['attributes']}
            self.assertEqual(attrs['service.name'], 'test-service')
            self.assertEqual(attrs['service.version'], '2.0.0')

    def test_send_otlp_handles_network_error(self):
        """Network errors are caught and logged."""
        with patch('urllib.request.urlopen', side_effect=OSError('Connection refused')):
            with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317'}):
                with patch.object(otel_export.logger, 'debug') as mock_debug:
                    otel_export._send_otlp('traces', {'test': 'data'})
                    # Should log debug but not raise
                    mock_debug.assert_called()

    def test_error_span_sets_status_code(self):
        """Error span status is set to code 2."""
        with patch.object(otel_export, '_send_otlp') as mock_send:
            with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317'}):
                otel_export.export_trace(
                    trace_id='a', span_id='b', parent_span_id='',
                    name='test', started_at='2026-09-16T14:30:00Z', duration_ms=1,
                    status='error', attributes={'error': 'ValueError'}
                )

            payload = mock_send.call_args[0][1]
            span = payload['resourceSpans'][0]['scopeSpans'][0]['spans'][0]
            self.assertEqual(span['status']['code'], 2)  # error

    def test_ok_span_sets_status_code(self):
        """OK span status is set to code 0."""
        with patch.object(otel_export, '_send_otlp') as mock_send:
            with patch.dict(os.environ, {'OTEL_EXPORTER_OTLP_ENDPOINT': 'http://localhost:4317'}):
                otel_export.export_trace(
                    trace_id='a', span_id='b', parent_span_id='',
                    name='test', started_at='2026-09-16T14:30:00Z', duration_ms=1,
                    status='ok', attributes={}
                )

            payload = mock_send.call_args[0][1]
            span = payload['resourceSpans'][0]['scopeSpans'][0]['spans'][0]
            self.assertEqual(span['status']['code'], 0)  # ok


class TestJsonFormatterOtel(unittest.TestCase):
    """Test that JSON formatter includes OTel headers."""

    def test_json_formatter_includes_otel_headers(self):
        """JSON logs include service name, version, environment."""
        from assistant import observability

        formatter = observability.JSONFormatter()
        record = observability.logger.makeRecord(
            'assistant', 20, 'test.py', 1, 'Test message', (), None
        )

        with patch.dict(os.environ, {
            'OTEL_SERVICE_NAME': 'test-app',
            'OTEL_SERVICE_VERSION': '1.0.0',
            'OTEL_DEPLOYMENT_ENVIRONMENT': 'test'
        }):
            # Reimport to pick up new env vars
            import importlib
            importlib.reload(observability)
            formatter = observability.JSONFormatter()

            result = formatter.format(record)
            payload = json.loads(result)
            self.assertEqual(payload['service.name'], 'test-app')
            self.assertEqual(payload['service.version'], '1.0.0')
            self.assertEqual(payload['deployment.environment'], 'test')


if __name__ == '__main__':
    unittest.main()
