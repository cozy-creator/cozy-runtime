A live preview never waits on the GPU: its VAE factors reach each device once from pinned memory, and a video's frames are picked as a view before anything is copied.
