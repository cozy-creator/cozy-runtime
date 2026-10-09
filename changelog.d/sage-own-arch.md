SageAttention2 builds only the kernel families `attention_upstream` calls on the card: SM90 skips upstream's SM89 family, SM89/SM120 skip its SM80 family. Other recipes' kernel keys are unchanged.
