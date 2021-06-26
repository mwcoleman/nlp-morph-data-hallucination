# show V's and X's on predicted output file according to gold file

def main():

    evaluate('/Users/roeeaharoni/git/morphological-reinflection/results/heb-task1-attention-exp-bugfix-II.external_eval.txt.test.predictions',
             '/Users/roeeaharoni/git/morphological-reinflection/data/heb/hebrew-task1-test',
             '/Users/roeeaharoni/git/morphological-reinflection/results/heb_attn_error_analysis.txt')

    # evaluate(
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/solutions/nfst/turkish-task1-solution',
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/biu/gold/turkish-task1-test',
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/turkish-task1-test-error_analysis.txt')
    #
    # evaluate(
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/solutions/nfst/russian-task1-solution',
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/biu/gold/russian-task1-test',
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/russian-task1-test-error_analysis.txt')

    # evaluate(
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_structured_finnish_results.txt.best.predictions',
    #     '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/finnish-task1-dev',
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_structured_finnish_results.txt.best.predictions.error_analysis')

    # compare_error_analysis('/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_structured_finnish_results.txt.best.predictions.error_analysis',
    #                        '/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_finnish_results.txt.best.predictions.error_analysis',
    #                        '/Users/roeeaharoni/GitHub/morphological-reinflection/results/error_analysis_finnish_joint_vs_joint_structured.txt')
    #
    # evaluate(
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/real-par-pos-russian-results.txt.best.predictions',
    #     '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/russian-task1-dev',
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_russian_results.txt.best.predictions.error_analysis')
    #
    # evaluate(
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_structured_russian_results.txt.best.predictions',
    #     '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/russian-task1-dev',
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_structured_russian_results.txt.best.predictions.error_analysis')
    #
    # compare_error_analysis(
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_structured_russian_results.txt.best.predictions.error_analysis',
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/joint_russian_results.txt.best.predictions.error_analysis',
    #     '/Users/roeeaharoni/GitHub/morphological-reinflection/results/error_analysis_russian_joint_vs_joint_structured.txt')

    return

    predicted_file_format = '/Users/roeeaharoni/GitHub/morphological-reinflection/results/\
joint_structured_{0}_results.txt.best.predictions'

    gold_file_format = '/Users/roeeaharoni/research_data/sigmorphon2016-master/data/{0}-task1-dev'

    output_file_format = '/Users/roeeaharoni/GitHub/morphological-reinflection/results/\
joint_structured_{0}_results.txt.best.predictions_error_analysis.txt'

    langs = ['arabic', 'finnish', 'georgian', 'russian', 'german', 'turkish', 'spanish', 'navajo']

    for lang in langs:
        predicted_file = predicted_file_format.format(lang)
        gold_file = gold_file_format.format(lang)
        output_file = output_file_format.format(lang)
        evaluate(predicted_file, gold_file, output_file)
        print 'created error analysis for {0} in: {1}'.format(lang, output_file)


def evaluate(predicted_file, gold_file, output_file):
    with open(predicted_file) as predicted:
        predicted_lines = predicted.readlines()
        with open(gold_file) as gold:
            gold_lines = gold.readlines()
            if not len(gold_lines) == len(predicted_lines):
                print 'file lengths mismatch, {0} lines in gold and {1} lines in prediction'.format(len(gold_lines),
                                                                                                    len(predicted_lines))

            else:
                with open(output_file, 'w') as output:
                    morph2results = {}
                    for i, predicted_line in enumerate(predicted_lines):

                        output_line = ''
                        [pred_lemma, pred_morph, predicted_inflection] = predicted_line.split()
                        [lemma, morph, gold_inflection] = gold_lines[i].split()

                        if pred_lemma != lemma:
                            print 'mismatch in index' + str(i)
                            return


                        if predicted_inflection == gold_inflection:
                            mark = 'V'
                        else:
                            mark = 'X'
                            line_format = "lemma:\n{0}\nfeatures:\n{1}\ngold:\n{2}\npredicted:\n{3}\n{4}\n"
                            output_line = line_format.format(lemma, morph, gold_inflection, predicted_inflection, mark)
                            if morph in morph2results:
                                morph2results[morph].append(output_line)
                            else:
                                morph2results[morph] = [output_line]
                        # output.write(output_line)


                    for morph in morph2results:
                        output.write('\n\n#################################\n\n')
                        for line in morph2results[morph]:
                            output.write(line)
                            output.write('\n')



def compare_error_analysis(error_file_1, error_file_2, output_file):

    only_ef1_errors = []
    only_ef2_errors = []
    with open(error_file_1) as ef1:
        ef1_lines = ef1.readlines()
        with open(error_file_2) as ef2:
            ef2_lines = ef2.readlines()
            intersection = set.intersection(set(ef1_lines), set(ef2_lines))
            for line in ef1_lines:
                if line not in intersection:
                    only_ef1_errors.append(line)
            for line in ef2_lines:
                if line not in intersection:
                    only_ef2_errors.append(line)

    with open(output_file, 'w') as output:
        lines = []
        lines.append('\nboth ({0}):\n===============\n'.format(len(intersection)))
        for l in intersection:
            lines.append(l)

        lines.append('\nonly in ' + error_file_1 + ' ({0}):\n===============\n'.format(len(only_ef1_errors)))
        v_lines, adj_lines, n_lines = group_by_pos(only_ef1_errors)
        lines.append('\nverb errors ({0})\n=======\n'.format(len(v_lines)))
        for l in v_lines:
            lines.append(l)
        lines.append('\nadj errors ({0})\n=======\n'.format(len(adj_lines)))
        for l in adj_lines:
            lines.append(l)
        lines.append('\nnoun errors ({0})\n=======\n'.format(len(n_lines)))
        for l in n_lines:
            lines.append(l)


        lines.append('\nonly in ' + error_file_2 + '({0}):\n===============\n'.format(len(only_ef2_errors)))
        v_lines, adj_lines, n_lines = group_by_pos(only_ef2_errors)
        lines.append('\nverb errors ({0})\n=======\n'.format(len(v_lines)))
        for l in v_lines:
            lines.append(l)
        lines.append('\nadj errors ({0})\n=======\n'.format(len(adj_lines)))
        for l in adj_lines:
            lines.append(l)
        lines.append('\nnoun errors ({0})\n=======\n'.format(len(n_lines)))
        for l in n_lines:
            lines.append(l)

        output.writelines(lines)
        print 'wrote comparison to {0}'.format(output_file)
    return


def group_by_pos(lines):
    v_lines = []
    adj_lines = []
    n_lines = []
    for line in lines:
        if 'pos=V' in line:
            v_lines.append(line)
        if 'pos=ADJ' in line:
            adj_lines.append(line)
        if 'pos=N' in line:
            n_lines.append(line)
    return v_lines, adj_lines, n_lines

if __name__ == '__main__':
    main()