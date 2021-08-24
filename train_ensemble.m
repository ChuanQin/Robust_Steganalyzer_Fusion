function train_ensemble(stego_feature_path, clf_path, randind)
    clf_dir = split(clf_path,'/'); clf_dir = join(clf_dir(1:end-1), '/'); clf_dir = clf_dir{1}; 
    if ~exist(clf_dir,'dir'); mkdir(clf_dir); end

    feature_dir = split(stego_feature_path,'/'); 
    feature_dir = join(feature_dir(1:end-1), '/'); 
    feature_dir = [feature_dir{1}, '/']; 
    cover_feature_path = [feature_dir, 'cover.mat'];

    if ~exist(cover_feature_path) || ~exist(stego_feature_path); fprintf('The feature mat files do not exist!\n'); exit; end
    cover = matfile(cover_feature_path); raw_names = cover.names; cover = cover.F;
    stego = matfile(stego_feature_path);

    % Sequencing the names
    names = cell(size(raw_names,1),1);
    for i=1:size(names,1); names{i} = strrep(raw_names(i,:),' ',''); end
    [names, idx] = sort(names);
    cover = cover(idx,:); stego = stego(idx,:);

    % Dividing training and testing datasets
    num_tst = 1000;
    trn_idx = randind(1:end-num_tst);
    tst_idx = randind(end-num_tst+1:end);

    % train ensemble classifier
    % settings.d_sub = 8;
    [esb_clf, ~] = ensemble_training(cover(trn_idx,:), stego(trn_idx,:));
    save(clf_path, 'esb_clf');

    test_result_cover = ensemble_testing(cover(tst_idx,:), esb_clf);
    test_result_stego = ensemble_testing(stego(tst_idx,:), esb_clf);

    % calc performance
    acc_cover = sum(test_result_cover.predictions==-1);
    acc_stego = sum(test_result_stego.predictions==+1);
    num_testing_samples = size(tst_idx,2);
    avg_acc = (acc_cover + acc_stego)/(num_testing_samples*2);
    acc_cover = acc_cover/num_testing_samples;
    acc_stego = acc_stego/num_testing_samples;

    % print
    fprintf([feature_dir, '\n']);
    fprintf(stego_feature_path);
    fprintf('Average Acc: %.4f\n',avg_acc);
    fprintf('Cover Acc: %.4f\n',acc_cover);
    fprintf('Stego Acc: %.4f\n',acc_stego);
    delete(gcp('nocreate'));
end