testo ──► normalizzazione ──► embedding ──► cosine vs label frasi
                                               │
                         score ≥ τ_alto ───────┴──► animazione frase intera
                                               │
                         score < τ_alto ───────┴──► spaCy (POS + lemma)
                                                      ──► riordino LIS
                                                      ──► lookup lemma → clip parola
                                                      ──► concatenazione + blending