import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer,BertTokenizer
from Models.DIAMIT.MM_model import MMmodel
from Models.IRENE.modeling_irene import IRENE
from Scripts.metrics import *

ROBERTA_PATH = '/ROBERTA/roberta_config'
BERT_PATH = '/BERT'

class Trainer(object):

    def __init__(self, datawrapper, model_parameters):
        self.datawrapper = datawrapper
        self.epochs = model_parameters['epochs']
        self.writer = SummaryWriter('./logs')

        self.device = self._get_device(model_parameters)
        self.tokenizer = self._get_tokenizer(model_parameters)
        self.lossfn = self._get_loss_fn(model_parameters)
        self.class_names = self._get_class_name(model_parameters)

        self.model_parameters = model_parameters

    def _get_class_name(self, model_parameters):
        return model_parameters['class_names']

    def _get_device(self, model_parameters):
        device = model_parameters['device']
        print("Running on:", device)
        return device

    def pad_zero(self, string, length):
        finalstrlist = []
        for index in range(len(string)):
            finalstr = string[index]
            for i in range(length-len(string[index])+100):
                finalstr += '0'
            finalstrlist.append(finalstr)
        return finalstrlist

    def _get_tokenizer(self, model_parameters):
        if model_parameters['report_model_name'] == 'bert-base-chinese':
            self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
        elif model_parameters['report_model_name'] == 'chinese-roberta-wwm-ext':
            self.tokenizer = BertTokenizer.from_pretrained(ROBERTA_PATH)
        else:
            print('unavailable name of report_model')
        return self.tokenizer

    def _get_loss_fn(self, model_parameters):
        if model_parameters['classes'] == 2:
            lossfn = nn.BCELoss()
        else:
            lossfn = nn.CrossEntropyLoss()
        return lossfn

    def _load_pre_trained_weights(self, model):
        try:
            load_state_dict = torch.load(self.model_parameters['pretrained_weights_path'])['state_dict']
            model.load_state_dict(load_state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def train(self, repeat_counts):

        # results
        fold_count = 0
        train_counts = []
        val_losses, val_accs, val_aucs, val_counts, val_epochs = [], [], [], [], []
        test_losses, test_accs, test_auc_mcs, test_aucs, test_cfrs, test_cfms, test_counts = [], [], [], [], [], [], []
        EPOCHTIMES = []

        # data loaders
        train_loader = self.datawrapper.get_data_loaders(is_train=0, droplast=self.model_parameters['train_droplast'],foldindex=0)
        valid_loader = self.datawrapper.get_data_loaders(is_train=1, droplast=self.model_parameters['val_droplast'],foldindex=0)
        test_loader = self.datawrapper.get_data_loaders(is_train=2, droplast=self.model_parameters['test_droplast'],foldindex=0)

        # repeat experiments
        for j in range(repeat_counts):
            print('EX: ', j)

            if self.model_parameters['MODEL'] == 'IRENE':
                model = IRENE(self.model_parameters['irene_config'], self.model_parameters['device'], self.model_parameters['W'], zero_head=True, vis=True, num_classes=self.model_parameters['classes']).to(self.device)
            else:
                model = MMmodel(
                     image_model_name = self.model_parameters['image_model_name'],
                     report_model_name = self.model_parameters['report_model_name'], # 'bert-base-chinese', 'chinese-roberta-wwm-ext'
                     img_aggregation_type = self.model_parameters['image_aggregation_type'], # 'AVG', '3D', 'DIAMIT'
                     img_weight_path= self.model_parameters['image_weights_path'],
                     input_W = self.model_parameters['W'], # width of slice
                     input_H = self.model_parameters['H'], # height of slice
                     input_D = self.model_parameters['D'], # slice number
                     multimodal_dim = self.model_parameters['mm_dim'],
                     mhsa_dim= self.model_parameters['mhsa_dim'],
                     mhsa_heads = self.model_parameters['mhsa_heads'],
                     dropout_rate = self.model_parameters['dropout_rate'],
                     mask_columns = self.model_parameters['mask_columns'],
                     bias = self.model_parameters['bias'],
                     channels = self.model_parameters['channels'],
                     nb_class = self.model_parameters['classes'],
                     freeze_layers = self.model_parameters['freeze_layers'],
                     predict_type = self.model_parameters['predict_type'], # 'img', 'rep', 'img+rep'
                     fusion_method = self.model_parameters['fusion_method'], # 'direct', 'proj'
                     device = self.device
                ).to(self.device)

            if self.model_parameters['pretrained_weights_path'] is not None:
                model = self._load_pre_trained_weights(model)

            # optimizer setting
            optimizer = torch.optim.Adam(model.parameters(), lr = self.model_parameters['lr_warmup'], weight_decay = self.model_parameters['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor = self.model_parameters['lr_reduce_factor'], patience = self.model_parameters['lr_reduce_patience'], min_lr=0, verbose=True)

            # Checkpoint folder
            model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

            # save config file
            save_config_file(model_checkpoints_folder)

            n_iter = 0
            valid_n_iter = 0

            # parameters for model saving
            best_valid_loss = np.inf
            best_acc = 0.0
            best_auc = 0.0
            best_epoch = 0
            val_count = []
            epoch_times = []
            best_model_path = ''

            print(f'Training...')

            for epoch_counter in range(self.model_parameters['epochs']):

                print(f'Epoch {epoch_counter}')
                time_start = time.time()
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_len = 0
                batch_count = 0
                train_count = [0 for i in range(self.model_parameters['classes'])]

                for xi, xr, xage, xsex, xl in tqdm(train_loader):
                    '''clear grad'''
                    optimizer.zero_grad()
                    xrt = self.tokenizer(text = list(xr),
                                         max_length = self.model_parameters['report_max_length'],
                                         return_tensors="pt",
                                         padding=True,
                                         truncation=True)
                    # padding
                    if self.model_parameters['padding']:
                        padding_length = self.model_parameters['report_max_length'] - xrt.data['input_ids'].shape[1]
                        if padding_length > 0:
                            paddings_inputids = torch.zeros((xrt.data['input_ids'].shape[0], padding_length), dtype=torch.int64)
                            paddings_tokentypeids = torch.zeros((xrt.data['input_ids'].shape[0], padding_length),dtype=torch.int64)
                            paddings_attentionmask = torch.zeros((xrt.data['input_ids'].shape[0], padding_length),dtype=torch.int64)
                            xrt.data['input_ids'] = torch.cat([xrt.data['input_ids'] ,paddings_inputids],dim=1)
                            xrt.data['token_type_ids'] = torch.cat([xrt.data['token_type_ids'], paddings_tokentypeids], dim=1)
                            xrt.data['attention_mask'] = torch.cat([xrt.data['attention_mask'], paddings_attentionmask], dim=1)
                    xrt = xrt.to(self.device)

                    '''To tensor'''
                    xi = xi.type(torch.FloatTensor)
                    xi = xi.to(self.device)

                    if self.model_parameters['classes'] == 2:
                         xl = xl.type(torch.FloatTensor)
                    else:
                        xl = xl.long()
                    xl = xl.to(self.device)

                    xage = xage.type(torch.FloatTensor)
                    xage = xage.to(self.device)

                    xsex = xsex.type(torch.FloatTensor)
                    xsex = xsex.to(self.device)

                    '''logit and loss'''
                    z, slice_atts = model(xi, xrt, xage, xsex)
                    if self.model_parameters['label_squeeze']:
                        xl = xl.squeeze(dim=1)
                    batch_loss = self.lossfn(z, xl)

                    epoch_loss += batch_loss.item()

                    batch_loss.backward()

                    optimizer.step()

                    '''save predict'''
                    batch_correct, batch_len,batch_label_count = 0, 0, []
                    if self.model_parameters['classes'] == 2:
                        batch_acc, batch_correct, batch_len, batch_label_count = acc_binary(z.cpu().detach(), xl)
                    elif self.model_parameters['classes'] > 2:
                        batch_acc, batch_correct, batch_len, batch_label_count = acc_mc(z.cpu().detach(), xl)
                    epoch_correct += batch_correct
                    epoch_len += batch_len
                    count_add(train_count, batch_label_count)

                    batch_count += 1

                    '''save to tensorboard'''
                    if n_iter % 2 == 0:
                        self.writer.add_scalar('train_loss_fold{0}'.format(fold_count), batch_loss, global_step=n_iter)

                    n_iter += 1
                # time
                epoch_time = time.time() - time_start
                print('Epoch time: {0}'.format(epoch_time))
                epoch_times.append(epoch_time)
                '''validate'''
                if epoch_counter % self.model_parameters['val_epoch'] == 0:
                    # predict
                    valid_loss, valid_acc, val_count, valid_auc, valid_aucs_mc = self._validate(model, valid_loader, n_iter)
                    print('Validate: ', j)
                    print('train distribute: ', end='')
                    for index in range(self.model_parameters['classes']):
                        print('{}:{} '.format(self.model_parameters['class_names'][index], train_count[index]), end='')
                    print()
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Epoch len: {3}\t'
                          'Train Loss:{4}\t'
                          'Train Acc:{5}\t'
                          'Valid Loss:{6}\t'
                          'Valid Acc:{7}\t'
                          'Valid Auc:{8}'.format(epoch_counter, batch_count, len(train_loader), epoch_len,
                                                 epoch_loss/batch_count, epoch_correct/epoch_len, valid_loss, valid_acc, valid_auc))
                    sys.stdout.flush()

                    # model save
                    if epoch_counter > self.model_parameters['save_epoch']:
                        if self.model_parameters['save_model_type'] == 'loss_acc':
                            '''loss - acc'''
                            if valid_loss < best_valid_loss:
                                best_valid_loss = valid_loss
                                best_auc = valid_auc
                                is_best = valid_acc > best_acc
                                if is_best:
                                    best_epoch = epoch_counter
                                    best_acc = valid_acc
                                    best_model_path = 'fold{0}epoch{1}acc{2}auc{3}loss{4}.pth.tar'.format(
                                        fold_count, epoch_counter,
                                        str(np.round(best_acc, 4)),
                                        str(np.round(valid_auc, 4)),
                                        str(np.round(best_valid_loss, 4)))
                        elif self.model_parameters['save_model_type'] == 'acc_loss':
                            '''acc - loss'''
                            if valid_acc > best_acc:
                                is_best = valid_loss < best_valid_loss
                                if is_best:
                                    best_acc = valid_acc
                                    best_auc = valid_auc
                                    best_epoch = epoch_counter
                                    best_valid_loss = valid_loss
                                    best_model_path = 'fold{0}epoch{1}acc{2}auc{3}loss{4}.pth.tar'.format(
                                                                       fold_count, epoch_counter,
                                                                       str(np.round(best_acc, 4)),
                                                                       str(np.round(valid_auc, 4)),
                                                                       str(np.round(best_valid_loss, 4)))
                        else:
                            print('save_model_type wrong value')

                    '''save to tensorboard'''
                    self.writer.add_scalar('validation_loss_fold{0}'.format(fold_count), valid_loss, global_step=valid_n_iter)
                    self.writer.add_scalar('validation_accuracy_fold{0}'.format(fold_count), valid_acc, global_step=valid_n_iter)
                    self.writer.add_scalar('validation_auc_fold{0}'.format(fold_count), valid_auc,global_step=valid_n_iter)
                    valid_n_iter += 1

                    '''warm up'''
                    if epoch_counter < self.model_parameters['warmup_epochs']:
                        step  = (self.model_parameters['lr'] - self.model_parameters['lr_warmup'])/self.model_parameters['warmup_epochs']
                        scheduler.optimizer.param_groups[0]['lr'] += step
                    else:
                        scheduler.step(valid_loss)
                    print('lr:', scheduler.optimizer.param_groups[0]['lr'])

            test_model_path = save_checkpoint(ck_root=self.model_parameters['checkpoint_path'],
                                              md_param=self.model_parameters,
                                              repeat_exp=j,
                                              epoch=best_epoch,
                                              state={
                                                  'epoch': best_epoch,
                                                  'state_dict': model.state_dict(),
                                                  'val_acc': best_acc,
                                                  'val_auc': best_auc,
                                                  'val_loss': best_valid_loss},
                                              modelpath=best_model_path)

            print('best epoch:{0} best val acc:{1}, best val auc:{2}'.format(best_epoch,best_acc, best_auc))
            print('Epoch avg time: {0}'.format(np.mean(epoch_times)))
            EPOCHTIMES.append(np.mean(epoch_times))

            ''' test '''
            test_loss, test_acc, test_auc_mc, test_auc,cf_result, cf_matrix, test_count = self._test(model, test_model_path, test_loader, testindex = j,
                                                                                                     savepath = self.model_parameters['checkpoint_path'] + '/test_{0}'.format(j) +
                                                                                                                '_'+self.model_parameters['dataset'] +
                                                                                                                '_'+self.model_parameters['image_aggregation_type'] + '/')

            ''' record '''
            train_counts.append(train_count)
            val_losses.append(best_valid_loss)
            val_accs.append(best_acc)
            val_aucs.append(best_auc)
            val_epochs.append(best_epoch)
            val_counts.append(val_count)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_auc_mcs.append(test_auc_mc)
            test_aucs.append(test_auc)
            test_cfrs.append(cf_result)
            test_cfms.append(cf_matrix)
            test_counts.append(test_count)
            fold_count += 1

        '''repeated experiments print'''
        for i in range(repeat_counts):
            print('EX:{0} train: '.format(i), end='')
            for index_t in range(self.model_parameters['classes']):
                print('{}:{} '.format(self.model_parameters['class_names'][index_t], train_counts[i][index_t]), end='')
            print(' val: ', end='')
            for index_v in range(self.model_parameters['classes']):
                print('{}:{} '.format(self.model_parameters['class_names'][index_v], val_counts[i][index_v]), end='')
            print(' test: ', end='')
            for index_te in range(self.model_parameters['classes']):
                print('{}:{} '.format(self.model_parameters['class_names'][index_te], test_counts[i][index_te]), end='')
            print("\nbest epoch: {0}\t val acc: {1}\tval auc:{2}\t "
                  "test loss: {3}\t test_acc: {4} test_auc: {5} auc_average:{6}\n"
                  "confusion matrix: {7}"
                  "confusion result: {8}"
                  "epoch time: {9}".format(val_epochs[i], val_accs[i], val_aucs[i],
                                                 test_losses[i], test_accs[i], test_auc_mcs[i],test_aucs[i],
                                                 test_cfms[i], test_cfrs[i], EPOCHTIMES[i]))
        print('{0} repeated experiments avg test acc: {1}, avg test auc: {2}, avg time per epoch{3}'.format(repeat_counts, np.mean(test_accs), np.mean(test_aucs), np.mean(EPOCHTIMES)))
        result_to_file(self.model_parameters['checkpoint_path'] + '/' + self.model_parameters['image_aggregation_type'] +
                       '_HW' + str(self.model_parameters['H']) + '_TXT' + str(self.model_parameters['report_max_length']) + '_MMDIM' + str(self.model_parameters['mm_dim'])
                       +'_lr' + str(self.model_parameters['lr']),
                       val_epochs, val_accs, val_aucs, test_losses, test_accs, test_auc_mcs, test_aucs, test_cfms,
                       test_cfrs, EPOCHTIMES)

    def _validate(self, model, valid_loader, n_iter):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            batch_index = 0
            epoch_len = 0
            epoch_correct = 0
            xls = []
            zs = []
            val_counts = [0 for i in range(self.model_parameters['classes'])]

            print(f'Validation step')
            for xi, xr, xage, xsex, xl in tqdm(valid_loader):
                xrt = self.tokenizer(text = list(xr),
                                     max_length = self.model_parameters['report_max_length'],
                                     return_tensors = "pt",
                                     padding = True,
                                     truncation = True)
                if self.model_parameters['padding']:
                    padding_length = self.model_parameters['report_max_length'] - xrt.data['input_ids'].shape[1]
                    if padding_length > 0:
                        paddings_inputids = torch.zeros((xrt.data['input_ids'].shape[0], padding_length), dtype=torch.int64)
                        paddings_tokentypeids = torch.zeros((xrt.data['input_ids'].shape[0], padding_length),
                                                            dtype=torch.int64)
                        paddings_attentionmask = torch.zeros((xrt.data['input_ids'].shape[0], padding_length),
                                                             dtype=torch.int64)
                        xrt.data['input_ids'] = torch.cat([xrt.data['input_ids'], paddings_inputids], dim=1)
                        xrt.data['token_type_ids'] = torch.cat([xrt.data['token_type_ids'], paddings_tokentypeids], dim=1)
                        xrt.data['attention_mask'] = torch.cat([xrt.data['attention_mask'], paddings_attentionmask], dim=1)
                '''To tensor'''
                xi = xi.type(torch.FloatTensor)
                xi = xi.to(self.device)

                if self.model_parameters['classes'] == 2:
                    xl = xl.type(torch.FloatTensor)
                else:
                    xl = xl.long()
                xl = xl.to(self.device)

                xage = xage.type(torch.FloatTensor)
                xage = xage.to(self.device)

                xsex = xsex.type(torch.FloatTensor)
                xsex = xsex.to(self.device)

                '''logit and loss'''
                z, slice_atts, contris = model(xi, xrt, xage, xsex)

                if self.model_parameters['label_squeeze'] :
                    xl = xl.squeeze(dim=1)
                batchloss = self.lossfn(z, xl)

                valid_loss += batchloss.item()

                xls.extend(xl.cpu().numpy())
                zs.extend(z.cpu().numpy())

                batch_correct, batch_len = 0, 0

                if self.model_parameters['classes'] == 2:
                    batch_acc, batch_correct, batch_len, batch_label_count = acc_binary(z.cpu().detach(), xl)
                elif self.model_parameters['classes'] > 2:
                    batch_acc, batch_correct, batch_len, batch_label_count = acc_mc(z.cpu().detach(), xl)
                else:
                    print('classes value error')

                epoch_correct += batch_correct
                epoch_len += batch_len
                count_add(val_counts, batch_label_count)

                batch_index += 1

            xls = np.array(xls)
            zs = np.array(zs)
            aucs, auc_ = [], 0
            if self.model_parameters['classes'] == 2:
                auc_, fpr, tpr, threshold, target_, predict_ = auc_cal_binary(xls, zs, self.class_names, is_draw=False, is_save=False)
            elif self.model_parameters['classes'] > 2:
                aucs, auc_, fpr_dict, tpr_dict, auc_dict, thresholds, target_binary, predict_ = auc_mc(xls, zs, self.class_names, is_draw=False, is_save=False)
            else:
                print('classes value error')

            valid_loss /= batch_index
            acc = epoch_correct / epoch_len

        print('val len:', epoch_len)
        print('val distribute: ', end='')
        for index in range(self.model_parameters['classes']):
            print('{}:{} '.format(self.model_parameters['class_names'][index], val_counts[index]), end='')
        print()

        model.train()

        return valid_loss, acc, val_counts, auc_, aucs

    def _test(self, model, modelpath, test_loader, testindex, savepath):
        load_model = torch.load(modelpath)
        model.load_state_dict(load_model['state_dict'])
        print('model path', modelpath)
        print("best model on {0} epoch with acc {1} auc {2} loss {3}".format(load_model['epoch'],
                                                                             load_model['val_acc'],
                                                                             load_model['val_auc'],
                                                                             load_model['val_loss']))

        print(f'Test step')
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            batch_index = 0
            epoch_len = 0
            epoch_correct = 0
            xls = []
            zs = []
            contris_all, subnames, xrs, slice_attsall, img_origins,reg_attsall = [], [], [], [], [], []
            test_counts = [0 for i in range(self.model_parameters['classes'])]

            print(f'Test step')
            for xi, xr, xage, xsex,xl, img_origin, subname in tqdm(test_loader):

                xrt = self.tokenizer(text=list(xr),
                                     max_length = self.model_parameters['report_max_length'],
                                     return_tensors="pt",
                                     padding=True,
                                     truncation=True)
                if self.model_parameters['padding']:
                    padding_length = self.model_parameters['report_max_length'] - xrt.data['input_ids'].shape[1]
                    if padding_length > 0:
                        paddings_inputids = torch.zeros((xrt.data['input_ids'].shape[0], padding_length), dtype=torch.int64)
                        paddings_tokentypeids = torch.zeros((xrt.data['input_ids'].shape[0], padding_length),
                                                            dtype=torch.int64)
                        paddings_attentionmask = torch.zeros((xrt.data['input_ids'].shape[0], padding_length),
                                                             dtype=torch.int64)
                        xrt.data['input_ids'] = torch.cat([xrt.data['input_ids'], paddings_inputids], dim=1)
                        xrt.data['token_type_ids'] = torch.cat([xrt.data['token_type_ids'], paddings_tokentypeids], dim=1)
                        xrt.data['attention_mask'] = torch.cat([xrt.data['attention_mask'], paddings_attentionmask], dim=1)
                '''To tensor'''
                xi = xi.type(torch.FloatTensor)
                xi = xi.to(self.device)

                if self.model_parameters['classes'] == 2:
                    xl = xl.type(torch.FloatTensor)
                else:
                    xl = xl.long()
                xl = xl.to(self.device)

                xage = xage.type(torch.FloatTensor)
                xage = xage.to(self.device)

                xsex = xsex.type(torch.FloatTensor)
                xsex = xsex.to(self.device)

                '''logit and loss'''
                contris = None

                start = time.perf_counter()
                z, slice_atts, contris = model(xi, xrt, xage, xsex)
                end = time.perf_counter()
                print('infer_time:', end-start)
                if self.model_parameters['label_squeeze']:
                    xl = xl.squeeze(dim=1)
                batch_loss = self.lossfn(z, xl)

                test_loss += batch_loss.item()

                xls.extend(xl.cpu().numpy())
                zs.extend(z.cpu().numpy())
                xrs.extend(xr)
                subnames.extend(subname)
                if len(img_origin) == 0:
                    img_origins.extend(xi.cpu().numpy())
                else:
                    img_origins.extend(img_origin.cpu().numpy())
                if self.model_parameters['MODEL'] == 'IRENE':
                    slice_attsall.append(slice_atts)
                else:
                    if slice_atts != None:
                        slice_attsall.extend(slice_atts.cpu().numpy())
                batch_correct, batch_len =0, 0

                if self.model_parameters['classes'] == 2:
                    batch_acc, batch_correct, batch_len, batch_label_count = acc_binary(z.cpu().detach(), xl)
                elif self.model_parameters['classes'] > 2:
                    batch_acc, batch_correct, batch_len, batch_label_count = acc_mc(z.cpu().detach(), xl)
                else:
                    print('classes value error')

                epoch_correct += batch_correct
                epoch_len += batch_len
                count_add(test_counts, batch_label_count)

                batch_index += 1

            xls_ = np.array(xls).tolist()
            zs_ = np.array(zs).tolist()
            contris_all = np.array(contris_all).tolist()
            img_origins = np.array(img_origins).tolist()
            if self.model_parameters['MODEL'] == 'IRENE':
                save_IRENE_results(savepath, zs_, xls_, xrs, slice_attsall, img_origins, subnames)
            else:
                slice_attsall = np.array(slice_attsall).tolist()
                save_test_results(savepath, zs_, xls_, contris_all, xrs, slice_attsall, img_origins, subnames)

            xls = np.array(xls)
            zs = np.array(zs)
            aucs, auc_ = [], 0
            if self.model_parameters['classes'] == 2:
                auc_, fpr, tpr, threshold, target_, predict_ = auc_cal_binary(xls, zs, self.class_names, is_draw=self.model_parameters['draw_roc'], is_save=self.model_parameters['save_auc'],ck_path=modelpath)
                cf, cfm = fm_binary(zs, xls,self.class_names)
            elif self.model_parameters['classes'] > 2:
                aucs, auc_, fpr_dict, tpr_dict, auc_dict, thresholds, target_binary, predict_ = auc_mc(xls, zs, self.class_names, is_draw = self.model_parameters['draw_roc'], is_save=self.model_parameters['save_auc'],ck_path=modelpath)
                cf, cfm = fm_mc(zs, xls, self.class_names)
            else:
                print('classes value error')

            test_loss /= batch_index
            test_acc = epoch_correct / epoch_len

            print('test len:', epoch_len)
            print('test distribute: ', end='')
            for index in range(self.model_parameters['classes']):
                print('{}:{} '.format(self.model_parameters['class_names'][index], test_counts[index]), end='')
            print()



            '''print'''
            print('Test on {0} image_text pairs:\n'
                  'Test loss on full epoch model:{1}\t'
                  'Test acc on full epoch model:{2}\t'
                  'Test auc on full epoch model:{3}\t'
                  .format(epoch_len, test_loss, test_acc, auc_))
            if self.model_parameters['classes'] > 2:
                print('Test auc :{0}\t'.format(aucs))

            return test_loss, test_acc, aucs, auc_, cf, cfm, test_counts