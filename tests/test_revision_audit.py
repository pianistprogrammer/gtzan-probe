import unittest
import numpy as np
from gtzan_probe.revision_audit import exact_top_mask, band_metadata

class AuditTests(unittest.TestCase):
    def test_tied_superpixels_have_exact_cardinality(self):
        x=np.ones((10,10))
        mask=exact_top_mask(x,.1)
        self.assertEqual(mask.sum(),10)
        np.testing.assert_array_equal(np.flatnonzero(mask),np.arange(10))
    def test_ranked_mask_selects_largest_magnitudes(self):
        x=np.array([[2.,8.],[4.,1.]])
        np.testing.assert_array_equal(exact_top_mask(x,.5),[[False,True],[True,False]])
    def test_invalid_input_not_silently_accepted(self):
        with self.assertRaises(ValueError):exact_top_mask([np.nan],.1)
        with self.assertRaises(ValueError):exact_top_mask([1],1.1)
    def test_band_support_respects_nyquist(self):
        b=band_metadata()
        self.assertEqual(b.start_bin_inclusive.tolist(),[0,10,30,80])
        self.assertEqual(b.stop_bin_exclusive.tolist(),[10,30,80,128])
        self.assertAlmostEqual(b.support_max_hz.iloc[-1],11025.)
        self.assertTrue((b.first_center_hz<b.last_center_hz).all())


class AudioFailureTests(unittest.TestCase):
    def test_unreadable_audio_raises_instead_of_becoming_silence(self):
        from unittest.mock import patch
        from gtzan_probe.dataset import GTZANDataset
        dataset=GTZANDataset.__new__(GTZANDataset)
        dataset._audio_cache={}
        with patch('gtzan_probe.dataset.resolve_audio_path',return_value='bad.wav'), patch('gtzan_probe.dataset.librosa.load',side_effect=OSError('decode failure')):
            with self.assertRaises(OSError):dataset._load_audio('bad.wav')
        self.assertEqual(dataset._audio_cache,{})

class FilterProbeTests(unittest.TestCase):
    def test_zero_gain_is_exact_sham_and_filter_has_targeted_effect(self):
        from scipy.signal import butter
        from gtzan_probe.filter_probe import intervene
        sr=22050;t=np.arange(sr)/sr
        y=np.sin(2*np.pi*80*t)+np.sin(2*np.pi*3000*t)
        sos=butter(4,250,fs=sr,output='sos')
        np.testing.assert_array_equal(intervene(y,sos,0),y)
        z=intervene(y,sos,-12)
        # Ignore edge transients; verify low component decreases while the high component is retained.
        section=slice(sr//10,-sr//10)
        def amplitude(x,hz):return abs(np.mean(x[section]*np.exp(-2j*np.pi*hz*t[section])))
        self.assertLess(amplitude(z,80)/amplitude(y,80),.3)
        self.assertGreater(amplitude(z,3000)/amplitude(y,3000),.99)

if __name__=='__main__':unittest.main()
