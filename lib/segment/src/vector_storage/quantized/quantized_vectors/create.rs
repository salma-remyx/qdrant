use std::path::Path;
use std::sync::atomic::AtomicBool;

use common::fs::atomic_save_json;
use common::generic_consts::{Random, Sequential};
use common::types::PointOffsetType;

use super::{
    QUANTIZED_CONFIG_PATH, QuantizedVectors, QuantizedVectorsConfig, QuantizedVectorsStorageType,
    is_query_rotation_required,
};
use crate::common::operation_error::{OperationError, OperationResult};
use crate::data_types::primitive::PrimitiveVectorElement;
use crate::data_types::vectors::VectorElementType;
use crate::types::{
    BinaryQuantization, Distance, MultiVectorConfig, ProductQuantization, QuantizationConfig,
    ScalarQuantization, TurboQuantization, VectorStorageDatatype,
};
use crate::vector_storage::quantized::quantized_multivector_storage::MultivectorOffset;
use crate::vector_storage::turbo::TurboVectorStorage;
use crate::vector_storage::turbo::multi::TurboMultiVectorStorage;
use crate::vector_storage::{
    DenseTQVectorStorage, DenseVectorStorage, MultiTQVectorStorage, MultiVectorStorage,
    VectorStorageEnum, VectorStorageRead,
};

impl QuantizedVectors {
    pub fn create(
        vector_storage: &VectorStorageEnum,
        quantization_config: &QuantizationConfig,
        storage_type: QuantizedVectorsStorageType,
        path: &Path,
        max_threads: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        match vector_storage {
            VectorStorageEnum::DenseVolatile(v) => Self::create_impl(
                v,
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            #[cfg(test)]
            VectorStorageEnum::DenseVolatileByte(v) => Self::create_impl(
                v,
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            #[cfg(test)]
            VectorStorageEnum::DenseVolatileHalf(v) => Self::create_impl(
                v,
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::DenseMemmap(v) => Self::create_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::DenseMemmapByte(v) => Self::create_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::DenseMemmapHalf(v) => Self::create_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            #[cfg(target_os = "linux")]
            VectorStorageEnum::DenseUring(v) => Self::create_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            #[cfg(target_os = "linux")]
            VectorStorageEnum::DenseUringByte(v) => Self::create_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            #[cfg(target_os = "linux")]
            VectorStorageEnum::DenseUringHalf(v) => Self::create_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::DenseAppendableMemmap(v) => Self::create_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::DenseAppendableMemmapByte(v) => Self::create_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::DenseAppendableMemmapHalf(v) => Self::create_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::DenseTurbo(v) => Self::create_turbo_impl(
                v,
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::MultiDenseTurbo(v) => Self::create_turbo_multi_impl(
                v,
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::SparseVolatile(_) => Err(OperationError::WrongSparse),
            VectorStorageEnum::SparseMmap(_) => Err(OperationError::WrongSparse),
            VectorStorageEnum::MultiDenseVolatile(v) => Self::create_multi_impl(
                v,
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            #[cfg(test)]
            VectorStorageEnum::MultiDenseVolatileByte(v) => Self::create_multi_impl(
                v,
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            #[cfg(test)]
            VectorStorageEnum::MultiDenseVolatileHalf(v) => Self::create_multi_impl(
                v,
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::MultiDenseAppendableMemmap(v) => Self::create_multi_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::MultiDenseAppendableMemmapByte(v) => Self::create_multi_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::MultiDenseAppendableMemmapHalf(v) => Self::create_multi_impl(
                v.as_ref(),
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::EmptyDense(v) => Self::create_impl(
                v,
                quantization_config,
                storage_type,
                path,
                max_threads,
                stopped,
            ),
            VectorStorageEnum::EmptySparse(_) => Err(OperationError::WrongSparse),
        }
    }

    fn create_impl<
        TElement: PrimitiveVectorElement,
        TVectorStorage: DenseVectorStorage<TElement> + Send + Sync,
    >(
        vector_storage: &TVectorStorage,
        quantization_config: &QuantizationConfig,
        storage_type: QuantizedVectorsStorageType,
        path: &Path,
        max_threads: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        let dim = vector_storage.vector_dim();
        let count = vector_storage.total_vector_count();
        let distance = vector_storage.distance();
        let datatype = vector_storage.datatype();
        let vectors = (0..count as PointOffsetType).map(|i| {
            let vector = vector_storage.get_dense::<Sequential>(i);
            PrimitiveVectorElement::quantization_preprocess(quantization_config, distance, vector)
        });
        let on_disk_vector_storage = vector_storage.is_on_disk();

        Self::quantize_dense(
            vectors,
            quantization_config,
            distance,
            datatype,
            false,
            dim,
            count,
            on_disk_vector_storage,
            storage_type,
            path,
            max_threads,
            stopped,
        )
    }

    fn create_turbo_impl(
        vector_storage: &TurboVectorStorage,
        quantization_config: &QuantizationConfig,
        storage_type: QuantizedVectorsStorageType,
        path: &Path,
        max_threads: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        let dim = vector_storage.vector_dim();
        let count = vector_storage.total_vector_count();
        let distance = vector_storage.distance();
        let on_disk_vector_storage = vector_storage.is_on_disk();

        let (datatype, rotate_query) =
            is_query_rotation_required(vector_storage.datatype(), distance);
        // `rotate_query` is exactly "rotation preserves this metric", i.e. the
        // vectors are kept rotated and queries are rotated to match.
        let keep_rotated = rotate_query;

        let vectors = (0..count as PointOffsetType)
            .map(move |i| vector_storage.get_dense_for_requantization(i, keep_rotated));

        Self::quantize_dense(
            vectors,
            quantization_config,
            distance,
            datatype,
            rotate_query,
            dim,
            count,
            on_disk_vector_storage,
            storage_type,
            path,
            max_threads,
            stopped,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn quantize_dense<'a>(
        vectors: impl Iterator<Item = impl AsRef<[VectorElementType]> + Send + Sync + 'a> + Clone + Send,
        quantization_config: &QuantizationConfig,
        distance: Distance,
        datatype: VectorStorageDatatype,
        rotate_query: bool,
        dim: usize,
        count: usize,
        on_disk_vector_storage: bool,
        storage_type: QuantizedVectorsStorageType,
        path: &Path,
        max_threads: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        let vector_parameters = Self::construct_vector_parameters(
            quantization_config,
            distance,
            dim,
            count,
            storage_type,
        );

        let quantized_storage = match quantization_config {
            QuantizationConfig::Scalar(ScalarQuantization {
                scalar: scalar_config,
            }) => Self::create_scalar(
                vectors,
                &vector_parameters,
                count,
                scalar_config,
                storage_type,
                path,
                on_disk_vector_storage,
                stopped,
            )?,
            QuantizationConfig::Product(ProductQuantization { product: pq_config }) => {
                Self::create_pq(
                    vectors,
                    &vector_parameters,
                    count,
                    pq_config,
                    storage_type,
                    path,
                    on_disk_vector_storage,
                    max_threads,
                    stopped,
                )?
            }
            QuantizationConfig::Binary(BinaryQuantization {
                binary: binary_config,
            }) => Self::create_binary(
                vectors,
                &vector_parameters,
                count,
                binary_config,
                storage_type,
                path,
                on_disk_vector_storage,
                stopped,
            )?,
            QuantizationConfig::Turbo(TurboQuantization {
                turbo: turbo_config,
            }) => Self::create_turbo(
                vectors,
                &vector_parameters,
                count,
                turbo_config,
                storage_type,
                path,
                on_disk_vector_storage,
                max_threads,
                stopped,
            )?,
        };

        let quantized_vectors_config = QuantizedVectorsConfig {
            quantization_config: quantization_config.clone(),
            vector_parameters,
            storage_type,
        };

        let quantized_vectors = QuantizedVectors {
            storage_impl: quantized_storage,
            config: quantized_vectors_config,
            path: path.to_path_buf(),
            distance,
            datatype,
            rotate_query,
        };

        atomic_save_json(&path.join(QUANTIZED_CONFIG_PATH), &quantized_vectors.config)?;
        Ok(quantized_vectors)
    }

    fn create_multi_impl<
        TElement: PrimitiveVectorElement + 'static,
        TVectorStorage: MultiVectorStorage<TElement> + Send + Sync,
    >(
        vector_storage: &TVectorStorage,
        quantization_config: &QuantizationConfig,
        storage_type: QuantizedVectorsStorageType,
        path: &Path,
        max_threads: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        let dim = vector_storage.vector_dim();
        let distance = vector_storage.distance();
        let datatype = vector_storage.datatype();
        let multi_vector_config = *vector_storage.multi_vector_config();
        let vectors = vector_storage.iterate_inner_vectors().map(|vector| {
            PrimitiveVectorElement::quantization_preprocess(quantization_config, distance, vector)
        });
        let inner_vectors_count = vectors.clone().count();
        let vectors_count = vector_storage.total_vector_count();
        let on_disk_vector_storage = vector_storage.is_on_disk();

        let offsets = (0..vectors_count as PointOffsetType)
            .map(|idx| {
                vector_storage
                    .get_multi::<Random>(idx)
                    .as_ref()
                    .vectors_count() as PointOffsetType
            })
            .scan(0, accumulate_offset);

        Self::quantize_multi(
            vectors,
            offsets,
            quantization_config,
            distance,
            datatype,
            false,
            dim,
            vectors_count,
            inner_vectors_count,
            multi_vector_config,
            on_disk_vector_storage,
            storage_type,
            path,
            max_threads,
            stopped,
        )
    }

    /// Multivector counterpart of [`Self::create_turbo_impl`].
    fn create_turbo_multi_impl(
        vector_storage: &TurboMultiVectorStorage,
        quantization_config: &QuantizationConfig,
        storage_type: QuantizedVectorsStorageType,
        path: &Path,
        max_threads: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        let dim = vector_storage.vector_dim();
        let distance = vector_storage.distance();
        let multi_vector_config = *vector_storage.multi_vector_config();
        let vectors_count = vector_storage.total_vector_count();
        let on_disk_vector_storage = vector_storage.is_on_disk();

        let (datatype, rotate_query) =
            is_query_rotation_required(vector_storage.datatype(), distance);
        let keep_rotated = rotate_query;

        let vectors = (0..vectors_count as PointOffsetType).flat_map(move |key| {
            vector_storage.get_inner_dense_for_requantization(key, keep_rotated)
        });
        let inner_vectors_count: usize = (0..vectors_count as PointOffsetType)
            .map(|key| vector_storage.point_inner_vectors_count(key))
            .sum();

        let offsets = (0..vectors_count as PointOffsetType)
            .map(|key| vector_storage.point_inner_vectors_count(key) as PointOffsetType)
            .scan(0, accumulate_offset);

        Self::quantize_multi(
            vectors,
            offsets,
            quantization_config,
            distance,
            datatype,
            rotate_query,
            dim,
            vectors_count,
            inner_vectors_count,
            multi_vector_config,
            on_disk_vector_storage,
            storage_type,
            path,
            max_threads,
            stopped,
        )
    }

    /// Shared tail of the multi-vector create paths. See [`Self::quantize_dense`].
    #[allow(clippy::too_many_arguments)]
    fn quantize_multi<'a>(
        vectors: impl Iterator<Item = impl AsRef<[VectorElementType]> + Send + Sync + 'a> + Clone + Send,
        offsets: impl Iterator<Item = MultivectorOffset>,
        quantization_config: &QuantizationConfig,
        distance: Distance,
        datatype: VectorStorageDatatype,
        rotate_query: bool,
        dim: usize,
        vectors_count: usize,
        inner_vectors_count: usize,
        multi_vector_config: MultiVectorConfig,
        on_disk_vector_storage: bool,
        storage_type: QuantizedVectorsStorageType,
        path: &Path,
        max_threads: usize,
        stopped: &AtomicBool,
    ) -> OperationResult<Self> {
        let vector_parameters = Self::construct_vector_parameters(
            quantization_config,
            distance,
            dim,
            inner_vectors_count,
            storage_type,
        );

        let quantized_storage = match quantization_config {
            QuantizationConfig::Scalar(ScalarQuantization {
                scalar: scalar_config,
            }) => Self::create_scalar_multi(
                vectors,
                offsets,
                &vector_parameters,
                vectors_count,
                inner_vectors_count,
                scalar_config,
                storage_type,
                multi_vector_config,
                path,
                on_disk_vector_storage,
                stopped,
            )?,
            QuantizationConfig::Product(ProductQuantization { product: pq_config }) => {
                Self::create_pq_multi(
                    vectors,
                    offsets,
                    &vector_parameters,
                    vectors_count,
                    inner_vectors_count,
                    pq_config,
                    storage_type,
                    multi_vector_config,
                    path,
                    on_disk_vector_storage,
                    max_threads,
                    stopped,
                )?
            }
            QuantizationConfig::Binary(BinaryQuantization {
                binary: binary_config,
            }) => Self::create_binary_multi(
                vectors,
                offsets,
                &vector_parameters,
                vectors_count,
                inner_vectors_count,
                binary_config,
                storage_type,
                multi_vector_config,
                path,
                on_disk_vector_storage,
                stopped,
            )?,
            QuantizationConfig::Turbo(TurboQuantization {
                turbo: turbo_config,
            }) => Self::create_turbo_multi(
                vectors,
                offsets,
                &vector_parameters,
                vectors_count,
                inner_vectors_count,
                turbo_config,
                storage_type,
                multi_vector_config,
                path,
                on_disk_vector_storage,
                max_threads,
                stopped,
            )?,
        };

        let quantized_vectors_config = QuantizedVectorsConfig {
            quantization_config: quantization_config.clone(),
            vector_parameters,
            storage_type,
        };

        let quantized_vectors = QuantizedVectors {
            storage_impl: quantized_storage,
            config: quantized_vectors_config,
            path: path.to_path_buf(),
            distance,
            datatype,
            rotate_query,
        };

        atomic_save_json(&path.join(QUANTIZED_CONFIG_PATH), &quantized_vectors.config)?;
        Ok(quantized_vectors)
    }
}

/// Running multivector offset accumulator for `Iterator::scan`: emits the start
/// offset for each multivector and advances by its inner-vector count.
// `scan` requires the callback to return `Option`, so the wrap is mandatory here.
#[allow(clippy::unnecessary_wraps)]
fn accumulate_offset(
    offset_acc: &mut PointOffsetType,
    multi_vector_len: PointOffsetType,
) -> Option<MultivectorOffset> {
    let offset = *offset_acc;
    *offset_acc += multi_vector_len;
    Some(MultivectorOffset {
        start: offset,
        count: multi_vector_len,
    })
}

#[cfg(test)]
mod tests {
    use common::counter::hardware_counter::HardwareCounterCell;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use tempfile::Builder;

    use super::*;
    use crate::data_types::vectors::{
        DenseVector, MultiDenseVectorInternal, QueryVector, TypedMultiDenseVectorRef,
    };
    use crate::types::{
        MultiVectorConfig, ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    };
    use crate::vector_storage::VectorStorage;
    use crate::vector_storage::turbo::multi::open_appendable_turbo_multi_vector_storage;
    use crate::vector_storage::turbo::open_appendable_turbo_vector_storage;

    /// Deterministic unit vectors so the metric preprocessing the scorer applies
    /// is a no-op and self-ranking is unambiguous.
    fn make_unit_vectors(dim: usize, count: usize, seed: u64) -> Vec<DenseVector> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                use rand::RngExt;
                let v: DenseVector = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
                let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
                v.iter().map(|&x| x / norm).collect()
            })
            .collect()
    }

    /// Building a scalar quantization straight from a Turbo source must score
    /// correctly: each stored vector ranks itself first against its own query.
    /// This exercises both the rotated-storage path (Dot/Cosine/Euclid keep the
    /// vectors rotated and rotate the query to match) and the rotate-back path
    /// (Manhattan).
    #[test]
    fn scalar_quantization_over_turbo_source_ranks_self_first() {
        const DIM: usize = 128;
        const COUNT: usize = 32;

        let quantization_config = QuantizationConfig::Scalar(ScalarQuantization {
            scalar: ScalarQuantizationConfig {
                r#type: ScalarType::Int8,
                quantile: None,
                always_ram: Some(true),
            },
        });

        for distance in [
            Distance::Dot,
            Distance::Cosine,
            Distance::Euclid,
            Distance::Manhattan,
        ] {
            let src_dir = Builder::new().prefix("turbo_src").tempdir().unwrap();
            let q_dir = Builder::new().prefix("turbo_quant").tempdir().unwrap();
            let hw = HardwareCounterCell::new();
            let stopped = AtomicBool::new(false);

            let inputs = make_unit_vectors(DIM, COUNT, 0xC0FFEE_u64.wrapping_add(distance as u64));
            let mut storage =
                open_appendable_turbo_vector_storage(src_dir.path(), DIM, distance, true).unwrap();
            for (i, v) in inputs.iter().enumerate() {
                storage
                    .insert_vector(i as PointOffsetType, v.as_slice().into(), &hw)
                    .unwrap();
            }

            let source = VectorStorageEnum::DenseTurbo(Box::new(storage));
            let quantized = QuantizedVectors::create(
                &source,
                &quantization_config,
                QuantizedVectorsStorageType::Immutable,
                q_dir.path(),
                1,
                &stopped,
            )
            .unwrap();

            for (q, query_vec) in inputs.iter().enumerate() {
                let scorer = quantized
                    .raw_scorer(
                        QueryVector::from(query_vec.clone()),
                        HardwareCounterCell::new(),
                    )
                    .unwrap();
                let scores: Vec<_> = (0..COUNT as PointOffsetType)
                    .map(|k| scorer.score_point(k))
                    .collect();
                let best = (0..COUNT)
                    .max_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap())
                    .unwrap();
                assert_eq!(
                    best, q,
                    "vector {q} not ranked first ({distance:?}): scores {scores:?}",
                );
            }
        }
    }

    /// Multivector counterpart: scalar quantization built straight from a Turbo
    /// multivector source must rank each point first against its own MaxSim query.
    #[test]
    fn scalar_quantization_over_turbo_multi_source_ranks_self_first() {
        const DIM: usize = 64;
        const COUNT: usize = 16;

        let quantization_config = QuantizationConfig::Scalar(ScalarQuantization {
            scalar: ScalarQuantizationConfig {
                r#type: ScalarType::Int8,
                quantile: None,
                always_ram: Some(true),
            },
        });

        for distance in [Distance::Dot, Distance::Cosine, Distance::Euclid] {
            let src_dir = Builder::new().prefix("turbo_multi_src").tempdir().unwrap();
            let q_dir = Builder::new()
                .prefix("turbo_multi_quant")
                .tempdir()
                .unwrap();
            let hw = HardwareCounterCell::new();
            let stopped = AtomicBool::new(false);

            // Each point gets `(i % 3) + 1` unit inner vectors.
            let inputs: Vec<MultiDenseVectorInternal> = (0..COUNT)
                .map(|i| {
                    let inner = (i % 3) + 1;
                    let flat: Vec<f32> = make_unit_vectors(DIM, inner, (i as u64) << 8)
                        .into_iter()
                        .flatten()
                        .collect();
                    MultiDenseVectorInternal::new(flat, DIM)
                })
                .collect();

            let mut storage = open_appendable_turbo_multi_vector_storage(
                src_dir.path(),
                DIM,
                distance,
                MultiVectorConfig::default(),
                true,
            )
            .unwrap();
            for (i, m) in inputs.iter().enumerate() {
                storage
                    .insert_vector(
                        i as PointOffsetType,
                        TypedMultiDenseVectorRef::from(m).into(),
                        &hw,
                    )
                    .unwrap();
            }

            let source = VectorStorageEnum::MultiDenseTurbo(Box::new(storage));
            let quantized = QuantizedVectors::create(
                &source,
                &quantization_config,
                QuantizedVectorsStorageType::Immutable,
                q_dir.path(),
                1,
                &stopped,
            )
            .unwrap();

            for (q, query_multi) in inputs.iter().enumerate() {
                let scorer = quantized
                    .raw_scorer(QueryVector::from(query_multi), HardwareCounterCell::new())
                    .unwrap();
                let scores: Vec<_> = (0..COUNT as PointOffsetType)
                    .map(|k| scorer.score_point(k))
                    .collect();
                let best = (0..COUNT)
                    .max_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap())
                    .unwrap();
                assert_eq!(
                    best, q,
                    "multi {q} not ranked first ({distance:?}): scores {scores:?}",
                );
            }
        }
    }
}
