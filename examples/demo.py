"""Fictional accounting dialogue; erroneous statements are deliberate."""
import argparse
from promptwork import HashingEmbedder, SentenceTransformerEmbedder, IGConfig, score_dialogue

TURNS = [
    {'role': 'user', 'content': 'Підготуй звіт для ФОП групи 3. Дохід за квартал 100000 гривень, без ПДВ.'},
    {'role': 'assistant', 'content': 'Для ФОП групи 3 без ПДВ застосуємо ставку 3%. Податок із доходу 100000 гривень становить 3000 гривень.'},
    {'role': 'user', 'content': 'Ставка 5%, а не 3%, згідно зі ст. 293 ПКУ. Виправ розрахунок податку.'},
    {'role': 'assistant', 'content': 'Виправляю: за ставкою 5% податок із доходу 100000 гривень становить 5000 гривень.'},
    {'role': 'user', 'content': 'Підготуй звіт для ФОП групи 3. Дохід за квартал 100000 гривень, без ПДВ.'},
    {'role': 'user', 'content': 'Дякую, ок'},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='SentenceTransformer model name (downloads may be required)')
    args = parser.parse_args()
    if args.model:
        embedder = SentenceTransformerEmbedder(model_name=args.model)
    else:
        print('WARNING: HashingEmbedder results are TOY-ONLY and not semantically meaningful.\n')
        embedder = HashingEmbedder()
    modes = ['none', 'first_user', 'previous_assistant']
    results = [score_dialogue(TURNS, embedder, IGConfig(user_relevance=mode)) for mode in modes]
    print('Fictional accounting example, not tax advice. Gains in nats.\n')
    print(f'{"turn / role":<19}' + ' | '.join(f'{m:^44}' for m in modes))
    print(f'{"":<19}' + ' | '.join(f'{"chunks  mean weight      IG    cumulative":^44}' for _ in modes))
    for i, source in enumerate(TURNS):
        cells = []
        for result in results:
            t = result.turns[i]
            mean = sum(t.weights) / t.n_chunks if t.n_chunks else 0
            cells.append(f'{t.n_chunks:6d} {mean:12.4f} {t.ig:9.4f} {t.cumulative_ig:12.4f}')
        print(f'{i}: {source["role"]:<16}' + ' | '.join(cells))
    print('\nUser share: ' + ' | '.join(f'{mode}={result.user_share:.3f}' for mode, result in zip(modes, results)))

    state_modes = ['both', 'user_only']
    state_results = [results[0], score_dialogue(TURNS, embedder, IGConfig(state_roles='user_only'))]
    print('\nUser-turn gain by state_roles (user_relevance=none):')
    print(f'{"turn":<8}' + ' | '.join(f'{mode:>12}' for mode in state_modes))
    for both_turn, user_turn in zip(state_results[0].turns, state_results[1].turns):
        if both_turn.role == 'user':
            print(f'{both_turn.index:<8}{both_turn.ig:12.4f} | {user_turn.ig:12.4f}')
    print('User share: ' + ' | '.join(
        f'{mode}={result.user_share:.3f}' for mode, result in zip(state_modes, state_results)))


if __name__ == '__main__':
    main()
