package producttest

import (
	"crypto/sha256"
	"fmt"
)

func digestOf(data []byte) string { return fmt.Sprintf("sha256:%x", sha256.Sum256(data)) }
