"""Failure-path scratch cleanup on synthetic arrays only."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.automl_sequence import inputs, modeling, reliance, workflow as w, test_workflow, test_reliance
from experiments.rigorous_transformer import comparison


class WorkflowMemoryTests(unittest.TestCase):
    def setUp(self):
        w.inputs,w.modeling,w.comparison=inputs,modeling,comparison
        self.arrays=[]

    def allocate(self,shape):
        with patch.object(modeling,'MEMORY_MAP_THRESHOLD',0):
            value=modeling.allocate_array(shape)
        value[:]=0
        self.arrays.append(value)
        return value

    def tearDown(self):
        for value in self.arrays: modeling.release_array(value)

    def assert_released(self):
        self.assertTrue(self.arrays)
        for value in self.arrays:
            self.assertTrue(value._mmap.closed)
            self.assertFalse(os.path.exists(value.filename))

    def test_compression_failure_removes_directory_before_return(self):
        fixture=test_workflow.WorkflowTest();fixture.setUp()
        self.addCleanup(w.release_inputs,fixture.data)
        paths=[]
        def fail(path,**arrays):
            paths.append(Path(path));Path(path).write_bytes(b'partial archive')
            raise OSError('synthetic disk failure')
        with patch.object(w.np,'savez_compressed',side_effect=fail):
            with self.assertRaisesRegex(OSError,'synthetic disk failure'):
                w.save_inputs(fixture.prefix,fixture.data,fixture.panel_manifest,fixture.config,fixture.impl)
        self.assertEqual(len(paths),1)
        self.assertFalse(paths[0].parent.exists())
        self.assertFalse(fixture.data['X_sequence']._mmap.closed)

    def test_temporal_shape_failure_releases_both_transforms(self):
        data=dict(indices={'train':np.arange(2)},x=np.zeros((2,12,2)),t=np.zeros((2,11,2)),
                  mask=np.ones((2,12),bool),tm=np.ones((2,11),bool))
        arm=dict(input='x',mask='mask',preparation='c',temporal_input='t',temporal_mask='tm',temporal_preparation='a')
        def transform(value,*args,**kwargs): return self.allocate(value.shape)
        with patch.object(modeling,'transform_inputs',side_effect=transform):
            with self.assertRaisesRegex(ValueError,'monthly shapes disagree'):
                w.arm_arrays(data,{'c':{},'a':{}},arm,'train')
        self.assertEqual(len(self.arrays),2);self.assert_released()

    def test_training_failure_releases_train_and_validation(self):
        import pandas as pd
        data=dict(metadata=pd.DataFrame({'PATIENT_ID':['a','b','c','d']}),manifest={},
                  indices={'train':np.array([0,1]),'validation':np.array([2,3])},y=np.array([0,1,0,1]))
        prep={};protocol=dict(preparation_hash=w.digest(prep),identity={'input_manifest_hash':w.digest({})},
            arms={'COUNT':{'family':'transformer'}},seeds=[42],model_config={})
        def arrays(*args,**kwargs): return self.allocate((2,12,2)),np.ones((2,12),bool),self.allocate((2,2))
        with patch.object(w,'arm_arrays',side_effect=arrays),patch.object(w,'table_exists',return_value=False),\
             patch.object(modeling,'train_encoder',side_effect=MemoryError('synthetic training failure')):
            with self.assertRaisesRegex(MemoryError,'synthetic training failure'):
                w.train_registered(data,prep,protocol,'SYNTHETIC')
        self.assertEqual(len(self.arrays),4);self.assert_released()

    def test_test_prediction_failure_releases_split_transforms(self):
        data=dict(manifest={},indices={'test':np.array([0,1])})
        prep={};protocol={'arms':{'COUNT':{'family':'transformer'}}}
        lock=dict(protocol_hash=w.digest(protocol),input_manifest_hash=w.digest({}),preparation_hash=w.digest(prep),
                  models={'COUNT':{'predictor_hashes':[{}]}})
        def arrays(*args,**kwargs): return self.allocate((2,12,2)),np.ones((2,12),bool),self.allocate((2,2))
        with patch.object(w,'arm_arrays',side_effect=arrays),patch.object(w,'table_exists',return_value=False),\
             patch.object(w,'read_artifacts',return_value={'lock.json':w.json_bytes(lock)}),\
             patch.object(modeling,'predict_encoder',side_effect=RuntimeError('synthetic prediction failure')):
            with self.assertRaisesRegex(RuntimeError,'synthetic prediction failure'):
                w.compare_run(data,prep,protocol,{'COUNT':[{'payload_hashes':{}}]},lock,'SYNTHETIC',evaluate_test=True)
        self.assert_released()

    def test_reliance_failure_releases_original_and_perturbed_maps(self):
        fixture=test_reliance.RelianceTests()
        data,prep,protocol,results,wf,model,_=fixture.fixture()
        protocol['arms']={'COUNT':protocol['arms']['COUNT']}
        original_arrays=wf.arm_arrays
        def arrays(*args,**kwargs):
            x,valid,static=original_arrays(*args,**kwargs)
            out=self.allocate(x.shape);out[:]=x
            return out,valid,static
        def take(value,rows):
            out=self.allocate((len(rows),)+value.shape[1:]);out[:]=value[rows]
            return out
        original_predict=model.predict_encoder
        calls=[]
        def predict(*args,**kwargs):
            calls.append(None)
            if len(calls)>len(protocol['seeds']): raise RuntimeError('synthetic permutation failure')
            return original_predict(*args,**kwargs)
        wf.arm_arrays=arrays;model.take_rows=take;model.predict_encoder=predict
        with self.assertRaisesRegex(RuntimeError,'synthetic permutation failure'):
            reliance.run(data,prep,protocol,results,wf,model,comparison)
        self.assertEqual(len(self.arrays),2);self.assert_released()


if __name__=='__main__': unittest.main()
