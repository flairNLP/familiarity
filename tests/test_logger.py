from pathlib import Path

from familarity.logger import setup_logger


def test_setup_logger(tmp_path: Path, capsys, caplog):
    output_path = tmp_path / "test_output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Call setup_logger with a temporary path
    logger = setup_logger(output_path)

    # Check if the log file path is correct
    log_file = output_path / "compute_metric.log"
    assert log_file.exists(), "Log file was not created."

    # Test writing a log message
    test_message = "This is a test log message."
    logger.info(test_message)

    # Flush the handlers to ensure all output is written
    for handler in logger.handlers:
        handler.flush()

    # Check if the log message was written to the file
    with open(log_file, "r") as f:
        log_content = f.read()
        assert test_message in log_content, "Log file does not contain the expected log message."

    # Check log capture with caplog as a fallback
    assert any(test_message in record.message for record in caplog.records), "Log message not found in caplog records."
