Return `ImageFile.from_image(image)` in an existing image result field to let the SDK encode and save final host pixels through its ordinary bounded output path. Defaults to lossless WebP, with PNG for sides over 16383 pixels; explicit PNG/WebP choices and alpha are preserved. `Outputs` remains available for streaming and intermediate publication.

Deferred encoders now enforce each returned field's existing `FileBound(max_bytes=...)` before writing its encoded blob, including legacy saved frames whose byte length is unknown during result validation.
