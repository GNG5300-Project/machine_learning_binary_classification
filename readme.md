bucket name: carbide-theme-428210-v5-kubeflowpipelines-default

curl -d '{
  "instances": [
    {
      "examples": {
        "b64": "Cp8DChgKBkxvYW5JRBIOCgwKCkkzOFBRVVFTOTYKDAoDQWdlEgUaAwoBOAoRCgZJbmNvbWUSBxoFCgPqnwUKFQoKTG9hbkFtb3VudBIHGgUKA5uLAwoVCgtDcmVkaXRTY29yZRIGGgQKAogEChcKDk1vbnRoc0VtcGxveWVkEgUaAwoBUAoXCg5OdW1DcmVkaXRMaW5lcxIFGgMKAQQKGAoMSW50ZXJlc3RSYXRlEggSBgoEFK5zQQoRCghMb2FuVGVybRIFGgMKASQKFAoIRFRJUmF0aW8SCBIGCgSuR+E+ChsKCUVkdWNhdGlvbhIOCgwKCkJhY2hlbG9yJ3MKHwoORW1wbG95bWVudFR5cGUSDQoLCglGdWxsLXRpbWUKHQoNTWFyaXRhbFN0YXR1cxIMCgoKCERpdm9yY2VkChYKC0hhc01vcnRnYWdlEgcKBQoDWWVzChgKDUhhc0RlcGVuZGVudHMSBwoFCgNZZXMKGAoLTG9hblB1cnBvc2USCQoHCgVPdGhlcgoWCgtIYXNDb1NpZ25lchIHCgUKA1llcw=="
      }
    }
  ]
}' -X POST http://localhost:8501/v1/models/1720308768:predict


python serve_with_tf_serving.py --port 8501 --model_name 1720308768 --model_path ./models/local/
