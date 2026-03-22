import ModelPerformance from '../components/ml/ModelPerformance';
import PredictionView from '../components/ml/PredictionView';
import FeatureImportance from '../components/ml/FeatureImportance';
import { useModels, useRetrainModel, usePredictions } from '../api/hooks';

export default function MLModels() {
  const { data: models, isLoading } = useModels();
  const { data: predictions } = usePredictions();
  const retrain = useRetrainModel();

  const activeModel = models?.find((m) => m.status === 'active');

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-gray-500">Loading models...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">ML Models</h1>
        <button
          onClick={() => {
            if (activeModel) retrain.mutate(activeModel.id);
          }}
          disabled={retrain.isPending || !activeModel}
          className="btn-primary text-sm disabled:opacity-50"
        >
          {retrain.isPending ? 'Retraining...' : 'Retrain Active Model'}
        </button>
      </div>

      {models && (
        <ModelPerformance
          models={models}
          onRetrain={(id) => retrain.mutate(id)}
          retraining={retrain.isPending}
        />
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {activeModel && (
          <FeatureImportance
            data={activeModel.feature_importance}
            modelName={activeModel.name}
          />
        )}
        {predictions && (
          <PredictionView predictions={predictions.slice(0, 9)} />
        )}
      </div>
    </div>
  );
}
