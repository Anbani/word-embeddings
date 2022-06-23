import sys
import struct
from tqdm import tqdm


def generate_tensors(model, model_name, folder_path="../../artifacts/web/"):
    vocab = model.wv.key_to_index
    num_rows = len(vocab)
    dim = model.vector_size

    tensor_bytes = f'{folder_path}{model_name}_tensors.bytes'
    tensor_tsv = f'{folder_path}{model_name}_tensors.tsv'
    labels = f'{folder_path}{model_name}_labels.tsv'

    tensor_bytes_file = open(tensor_bytes, 'wb')
    labels_file = open(labels, 'w', encoding='utf-8')
    tensor_tsv_file = open(tensor_tsv, 'w', encoding='utf-8')

    labels_file.write('word\tcount\n')

    pbar = tqdm(enumerate(vocab), total=num_rows)

    for i, word in pbar:
        if i % 1000 == 0:
            pbar.set_postfix_str(f"{word}")

        floatvals = model.wv[word].tolist()

        assert dim == len(floatvals)
        assert '\t' not in word
        assert word != None
        assert word != ''

        if i > 0:
            tensor_tsv_file.write('\n')

        for f_i, f in enumerate(floatvals):
            tensor_bytes_file.write(struct.pack('<f', f))

            if f_i > 0:
                tensor_tsv_file.write('\t%.8f' % f)
            else:
                tensor_tsv_file.write('%.8f' % f)

        # labels_file.write(('%s\t%s\n' % (word.lower(), model.wv.get_vecattr(word, "count"))).encode('utf-8'))
        labels_file.write(
            f'{word.lower()}\t{model.wv.get_vecattr(word, "count")}\n')

    print('\n')
    sys.stderr.flush()
    tensor_bytes_file.close()
    tensor_tsv_file.close()
    labels_file.close()

    print('''{
    "embeddings": [
      {
        "tensorName": %s,
        "tensorShape": [%d, %d],
        "tensorPath": "%s",
        "metadataPath": "%s"
      }
    ],
  }''' % (model_name, num_rows, dim, tensor_bytes, labels))


# Generate Tensorflow Projector-friendly tsv word2vec format
# def generate_tensors_text(model, tensor_filename, binary=False):
#     outfiletsv = tensor_filename + '_tensor.tsv'
#     outfiletsvmeta = tensor_filename + '_metadata.tsv'

#     with open(outfiletsv, 'w+', encoding='utf-8') as file_vector:
#         with open(outfiletsvmeta, 'w+', encoding='utf-8') as file_metadata:
#             file_metadata.write('word\tcount\n')
#             for word in tqdm(model.wv.index_to_key):
#                 wordstring = gensim.utils.to_utf8(word).decode("utf-8")
#                 countstring = str(model.wv.get_vecattr(word, "count"))
#                 file_metadata.write(wordstring + '\t' + countstring + '\n')
#                 vector_row = '\t'.join(str(x) for x in model.wv[word])
#                 file_vector.write(vector_row + '\n')

