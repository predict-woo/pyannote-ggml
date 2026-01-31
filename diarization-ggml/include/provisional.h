#pragma once

namespace diarization {

void provisional_init();
void provisional_update(const float* embeddings, int num_embeddings);
void provisional_finalize();

}
