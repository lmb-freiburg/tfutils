#
#  tfutils - A set of tools for training networks with tensorflow
#  Copyright (C) 2017  Benjamin Ummenhofer
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from .trainerbase import TrainerBase
from .simpletrainer import SimpleTrainer
from .helpers import *
import os
import operator

class EvolutionTrainer:
    """
    Evolution trainer supports training for networks with multiple evolutions (aka stages, levels, phases)

    This class internally uses the SimpleTrainer for the actual mainloop.
    """

    class _Evo:
        """Helper class that allows safe comparisons with evo strings"""
        def __init__(self,evolution, all_evolutions):
            self._evolution = evolution
            self._all_evolutions = all_evolutions

        def evo_compare(self, compare_op, other):
            """Compares this evolution with the specified other evo based on the index.
            
            compare_op: function
                Comparison operator for comparing the indices of the current evo and the given evo

            other: str
                The second argument for the comparison
                  
            Returns compare_op(evolution, other)
            """
            if not other in self._all_evolutions:
                raise ValueError('Unknown evolution "{0}"'.format(other))
            other_index = self._all_evolutions.index(other)
            evo_index = self._all_evolutions.index(self._evolution)
            return compare_op(evo_index, other_index)

        def __eq__(self,evo):
            return self.evo_compare(operator.eq, evo)
        
        def __ne__(self,evo):
            return self.evo_compare(operator.ne, evo)
        
        def __lt__(self,evo):
            return self.evo_compare(operator.lt, evo)
        
        def __le__(self,evo):
            return self.evo_compare(operator.le, evo)
        
        def __gt__(self,evo):
            return self.evo_compare(operator.gt, evo)
        
        def __ge__(self,evo):
            return self.evo_compare(operator.ge, evo) 

        def __str__(self):
            return self._evolution
        
        def name(self):
            return self._evolution


    def __init__(self, session, train_dir, evolutions, force_evo=None ):
        """
        session: tf.Session
            The tensorflow session for training

        train_dir: str
            Directory used for storing the individual evolutions

        evolutions: list of str
            List of evolutions. Each evolution string must be a valid and 
            unique directory name.

        force_evo: str
            Forces the current evolution to this value
            

        """
        self._check_evolution_names(evolutions)

        self._session = session
        self._train_dir = train_dir
        if isinstance(evolutions,tuple):
            self._evolutions = list(evolutions)
        elif isinstance(evolutions,list):
            self._evolutions = evolutions
        else:
            raise ValueError('invalid argument for "evolutions"')

        if len(self._evolutions) == 0:
            raise ValueError('evolution list is empty!')

        if not force_evo is None and not force_evo in self._evolutions:
            raise ValueError('{0} is not in the evolution list'.format(force_evo))


        current_evo, snapshot = self._retrieve_current_evolution_and_last_snapshot(force_evo)
        self._current_evolution = current_evo
        self._init_snapshot = snapshot
        self._current_evolution_dir = os.path.join(self._train_dir, self._current_evolution)

        if self._evolutions.index(self._current_evolution) > 0 and self._init_snapshot is None:
            raise RuntimeError('"{0}" is not the first evolution and no init snapshot was found for the previous evolution'.format(self._current_evolution))

        self._simpletrainer = SimpleTrainer(self._session, self._current_evolution_dir)


    @staticmethod
    def _check_evolution_names(evolutions):
        """Raises an error if something is wrong with the evolution names"""
        invalid_names = (
                TrainerBase.PROCESSID_FILE,
                TrainerBase.TRAIN_LOGDIR,
                TrainerBase.CHECKPOINTS_DIR,
                TrainerBase.RECOVERY_CHECKPOINTS_DIR,
                'testlogs',
                )
        for evo in evolutions:
            if evo in invalid_names:
                raise ValueError('"{0}" is not a valid evo name!'.format(evo))
        evo_set = set()
        for evo in evolutions:
            if not evo in evo_set:
                evo_set.add(evo)
            else:
                raise ValueError('There are at least two evolutions named "{0}"!'.format(evo))


    def _retrieve_current_evolution_and_last_snapshot(self, force_evo):
        """Returns the current evolution and a last snapshot tuple with
        evolution, iteration and path.

        force_evo: str
            Forces the current evo to be this evo
        """
        current_evo = None
        previous_evo = None
        if force_evo is None:
            for evo in reversed(self._evolutions):
                path = os.path.join(self._train_dir, evo)
                if os.path.isdir(path):
                    current_evo = evo
                    break
        else:
            current_evo = force_evo

        # set to first evo if there are no evo directories
        if current_evo is None:
            current_evo = self._evolutions[0]

        current_evo_index = self._evolutions.index(current_evo)
        if current_evo_index > 0:
            previous_evo = self._evolutions[current_evo_index-1]

        # try getting the snapshot from the current evolution
        snapshot = None
        if current_evo:
            evo_checkpoint_path = os.path.join(self._train_dir, current_evo, TrainerBase.CHECKPOINTS_DIR, TrainerBase.CHECKPOINTS_FILE_PREFIX)
            evo_recovery_checkpoint_path = os.path.join(self._train_dir, current_evo, TrainerBase.RECOVERY_CHECKPOINTS_DIR, TrainerBase.CHECKPOINTS_FILE_PREFIX)
            all_checkpoints = sorted(retrieve_all_checkpoints(evo_checkpoint_path) + retrieve_all_checkpoints(evo_recovery_checkpoint_path))
            if len(all_checkpoints):
                snapshot = (current_evo,) + all_checkpoints[-1]

        # try getting the snapshot from the previous evolution
        if previous_evo and snapshot is None:
            evo_checkpoint_path = os.path.join(self._train_dir, previous_evo, TrainerBase.CHECKPOINTS_DIR, TrainerBase.CHECKPOINTS_FILE_PREFIX)
            all_checkpoints = retrieve_all_checkpoints(evo_checkpoint_path)
            if len(all_checkpoints):
                snapshot = (previous_evo,) + all_checkpoints[-1]

        return current_evo, snapshot


    def session(self):
        """Returns the session for the trainer"""
        return self._session


    def coordinator(self):
        """Returns the coordinator for the trainer"""
        return self._simpletrainer.coordinator()


    def global_step(self):
        """Returns the tensor for the global step variable"""
        return self._simpletrainer.global_step()

    @property
    def current_evo(self):
        """Returns the current evolution as an object which allows comparisons to strings"""
        return EvolutionTrainer._Evo(self._current_evolution, self._evolutions)

    def init_snapshot(self):
        """Returns the tuple (evolution, iteration, snapshot_path).
        The snapshot path will be used as default in load_checkpoint().
        """
        return self._init_snapshot

    # some evo comparison shortcuts
    def current_evo_lt(self, evo):
        return EvolutionTrainer._Evo(self._current_evolution, self._evolutions) < evo

    def current_evo_le(self, evo):
        return EvolutionTrainer._Evo(self._current_evolution, self._evolutions) <= evo

    def current_evo_gt(self, evo):
        return EvolutionTrainer._Evo(self._current_evolution, self._evolutions) > evo

    def current_evo_ge(self, evo):
        return EvolutionTrainer._Evo(self._current_evolution, self._evolutions) >= evo

    def current_evo_eq(self, evo):
        return EvolutionTrainer._Evo(self._current_evolution, self._evolutions) == evo

    def current_evo_ne(self, evo):
        return EvolutionTrainer._Evo(self._current_evolution, self._evolutions) != evo

    def current_evo_in(self, evos):
        for evo in evos:
            if not evo in self._evolutions:
                raise ValueError('Unknown evolution "{0}"'.format(evo))
        return self._current_evolution in evos


    def load_checkpoint(self, checkpoint_filepath=None, verbose=True, remove_nonfinite_checkpoints=False):
        """Restores variables from a checkpoint file.

        checkpoint_filepath: str
            The path to the checkpoint file.
            If None then the last saved checkpoint will be loaded.

        verbose: bool
            If True prints which variables will be restored or skipped
        
        remove_nonfinite_checkpoints: bool
            If True a checkpoint which contains nonfinite values will be removed 
            before raising an exception.
            This option has not effect if checkpoint_filepath is given.
        """
        if checkpoint_filepath:
            print('loading', checkpoint_filepath, flush=True)
            optimistic_restore(self._session, checkpoint_filepath, verbose=verbose)
        else:
            if not self._init_snapshot is None:

                if self._current_evolution != self._init_snapshot[0]:
                    ignore_vars = ('global_step',)
                else:
                    ignore_vars = None

                last_checkpoint = self._init_snapshot[2]
                print('loading', last_checkpoint, flush=True)
                optimistic_restore(self._session, last_checkpoint, ignore_vars=ignore_vars, verbose=verbose, remove_nonfinite_checkpoints=remove_nonfinite_checkpoints)
            else:
                print('nothing to restore. no checkpoint found.', flush=True)



    def mainloop(self, *args, **kwargs):
        """Standard main loop

        max_iter: int
            Maximum iteraion number.

        train_ops: list of ops
            List of training ops.

        saver_interval: int
            Number of iterations between checkpoints.

        saver_max_to_keep: int
            Maximum number of snaphots to keep.

        saver_var_list: list or dict
            A list of variables to save or a dictionary which maps names to variables.
            This parameter is directly passed to the tf.train.Saver
            The list or dict must contain the global_step.
            If None a default list with the global_step and all trainable variables 
            will be created.

        recovery_saver_interval: float
            Time in minutes after last checkpoint which triggers saving a recovery checkpoint.

        summary_int_ops: list of tuple
            List of interval and operation tuples.
            E.g. [(100, summary1_op), (200, summary2_op)]

        display_interval: int
            Interval for running running display operations specified in 'display_str_ops'.

        display_str_ops: list of tuple
            List of string and operation tuples.
            E.g. [('MyLoss', op1), ('Ratio', op2)]

        test_int_fn: list of tuple
            List of interval and callable objects.
            E.g. [(1000, my_test_fn1), (1000, my_test_fn2)]
            The functions will be called before running the training ops

        runstats_interval: int
            Interval for logging cpu/mem usage and iterations per second

        trace_interval: int
            Interval for writing a trace snapshot.

        stop_time: 
            stop time in seconds since epoch.

        
        Returns a status code indicating if training was finished, crashed etc.
        """
        print('train evolution "{0}" ...'.format(self._current_evolution), flush=True)
        status = self._simpletrainer.mainloop(*args, **kwargs)

        evo_index = self._evolutions.index(self._current_evolution)

        # create the directory for the next evolution if training is finished.
        # This way a new EvolutionTrainer object will proceed with the next evolution.
        if status == TrainerBase.STATUS_TRAINING_FINISHED:
            print('training finished for evolution "{0}"'.format(self._current_evolution), flush=True)
            if evo_index + 1 < len(self._evolutions):
                next_evolution_dir = os.path.join(self._train_dir, self._evolutions[evo_index+1])
                os.makedirs(next_evolution_dir, exist_ok=True)
                status = TrainerBase.STATUS_TRAINING_UNFINISHED

        elif status == TrainerBase.STATUS_TRAINING_NAN_LOSS:
            pass
        else:
            pass

        return status

