---
description: 
---

---
description: Debug remote OpenAI-compatible LLM server connectivity
---

1. Check `BASE_URL`
2. Check whether proxy variables are interfering
3. Verify `/v1/models`
4. Verify exact model IDs returned by server
5. Compare requested model name with actual server model ID
6. On HTTP errors, print response body before raising