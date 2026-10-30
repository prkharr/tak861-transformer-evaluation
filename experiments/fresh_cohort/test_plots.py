import io
import unittest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from experiments.fresh_cohort import plots


class PlotTests(unittest.TestCase):
    def test_learning_curves_and_test_gate_render_real_metrics(self):
        results={'COUNT':[dict(identity={'seed':42},summary={'best_epoch':1},history=[dict(epoch=1,
            train_lift=3,validation_lift=2,train_log_loss=.3,validation_log_loss=.6)])]}
        fig=plots.training_curves(results)
        self.assertEqual(fig.axes[0].lines[0].get_ydata().tolist(),[3])
        self.assertEqual(fig.axes[0].lines[1].get_ydata().tolist(),[2])
        out=io.BytesIO(); FigureCanvasAgg(fig).print_png(out)
        self.assertGreater(len(out.getvalue()),1000)
        lock={'metrics':{'COUNT':{'train':{'primary_lift':3},'validation':{'primary_lift':2}}}}
        fig=plots.split_lifts(lock)
        self.assertEqual([p.get_height() for p in fig.axes[0].patches],[3,2])
        with self.assertRaises(ValueError):
            plots.split_lifts(lock,{'split':'validation','test_prediction_performed':False})
        fig=plots.split_lifts(lock,{'split':'test','test_prediction_performed':True,'metrics':{'COUNT':{'primary_lift':1.5}}})
        self.assertEqual([p.get_height() for p in fig.axes[0].patches],[3,2,1.5])


if __name__=='__main__':
    unittest.main()
