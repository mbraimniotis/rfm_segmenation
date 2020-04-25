"""
@author: Marios Braimniotis
"""

import matplotlib.pyplot as plt
import pandas as pd


def calculate_rfm():
    """
    Calculates the RFM score for the passengers.
    Outputs the RFM score to a csv file.
    Returns the RFM segmentation.
    """
    ride = pd.read_csv('ride.csv') # open ride.csv file
    fare = pd.read_csv('fares.csv') # open fares.csv file

    ## join ride.csv with fares.csv and filter based on rides actual revenue
    merged = pd.merge(ride, fare, on='id_request', how='inner')
    merged = merged[['id_request', 'id_passenger', 'actual_revenue', 'created_at']]
    merged['created_at'] = pd.to_datetime(merged['created_at']).dt.date
    merged = merged[merged.actual_revenue > 0]

    ## find max created_at date of ride which will be used for Recency(time since last ride) calculation
    max_date = merged.created_at.max()

    rfm_table = merged.groupby('id_passenger').agg(
        {
            'created_at': lambda x: (max_date - x.max()).days, # Recency - Time since last completed ride
            'id_request': lambda x: len(x), # Frequency - Total number of rides
            'actual_revenue': lambda x: x.sum() # Monetary Value - Total actual revenue
        }
    )

    rfm_table['created_at'] = rfm_table['created_at'].astype(int)
    rfm_table.rename(
        columns=
        {
            'created_at': 'recency',
            'id_request': 'frequency',
            'actual_revenue': 'monetary_value'
        }, inplace=True
    )

    quantiles = rfm_table.quantile(q=[0.25, 0.5, 0.75]) # define quantiles
    quantiles = quantiles.to_dict()
    ## apply segmentation functions to define rfm classification code for passengers
    rfm_segmentation = rfm_table
    rfm_segmentation['R_Quartile'] = rfm_segmentation['recency'].apply(
        r_class, args=('recency', quantiles,)
    )
    rfm_segmentation['F_Quartile'] = rfm_segmentation['frequency'].apply(
        f_m_class, args=('frequency', quantiles,)
    )
    rfm_segmentation['M_Quartile'] = rfm_segmentation['monetary_value'].apply(
        f_m_class, args=('monetary_value', quantiles,)
    )
    rfm_segmentation['RFMClass'] = (
        rfm_segmentation.R_Quartile.map(str) +
        rfm_segmentation.F_Quartile.map(str) +
        rfm_segmentation.M_Quartile.map(str)
    )
    ## define to segment name
    segt_map = {
        r'111': 'Best Passenger',
        r'311': 'Almost Lost Passenger',
        r'411': 'Lost Passenger',
        r'444': 'Lost Cheap Passenger',
        r'1[1-4][1-4]': 'Recent Passenger',
        r'[1-4]1[1-4]': 'Loyal Passenger',
        r'[1-4][1-4]1': 'Big Spenders',
        r'[1-4]+': 'Other'
    }
    ## assign to segment, based on rfm classification code
    rfm_segmentation['Segment'] = rfm_segmentation['RFMClass']
    rfm_segmentation['Segment'] = rfm_segmentation['Segment'].replace(segt_map, regex=True)
    rfm_segmentation.to_csv('rfm_output2.csv', sep=',')

    return rfm_segmentation


def plot_distribution_over_r_f(rfm_segmentation):
    """
    Plots the distribution of passengers over R and F
    """
    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 12), dpi=80)

    for i, p in enumerate(['R_Quartile', 'F_Quartile', 'M_Quartile']):
        parameters = {'R_Quartile':'Recency', 'F_Quartile':'Frequency', 'M_Quartile':'Revenue'}
        y = rfm_segmentation[p].value_counts().sort_index()
        x = y.index
        ax = axes[i]
        bars = ax.bar(x, y, color='silver')
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False)
        ax.set_title(
            f'Distribution of {parameters[p]}',
            fontsize=14
        )
        for _bar in bars:
            value = _bar.get_height()
            if value == y.max():
                _bar.set_color('firebrick')
            ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                value - 5,
                f'{int(value)}\n({int(value * 100 / y.sum())}%)',
                ha='center',
                va='top',
                color='w'
            )
    plt.savefig('D_1_RFM_Distribution.png')
    plt.show()


def count_number_of_customers_for_each_segment(rfm_segmentation):
    """
    Counts the number of passengers in each segment
    """
    segments_counts = rfm_segmentation['Segment'].value_counts().drop(labels=['Other']).sort_values(ascending=True)

    _, ax = plt.subplots(figsize=(16, 12), dpi=80)

    bars = ax.barh(
        range(len(segments_counts)),
        segments_counts,
        color='silver'
    )
    ax.set_frame_on(False)
    ax.tick_params(
        left=False,
        bottom=False,
        labelbottom=False
    )
    ax.set_yticks(range(len(segments_counts)))
    ax.set_yticklabels(segments_counts.index)

    for i, _bar in enumerate(bars):
        value = _bar.get_width()
        if segments_counts.index[i] in ['Best Passenger', 'Big Spenders', 'Loyal Passenger']:
            _bar.set_color('firebrick')
        ax.text(
            value,
            _bar.get_y() + _bar.get_height()/2,
            f'{int(value)} ({int(value*100/segments_counts.sum())}%)',
            va='center',
            ha='left'
        )
    plt.savefig('D_1_Passengers_Distribution2.png')
    plt.show()


"""
We create two functions for the RFM segmentation since, being high recency is bad,
while high frequency and monetary value is good.
"""
def r_class(value, param, quartiles_dict):
    """
    Arguments (value, param = recency, quartiles_dict = quartiles dict)
    """
    if value <= quartiles_dict[param][0.25]:
        return 1
    elif value <= quartiles_dict[param][0.50]:
        return 2
    elif value <= quartiles_dict[param][0.75]:
        return 3
    else:
        return 4


def f_m_class(value, param, quartiles_dict):
    """
    # Arguments (value, param = monetary_value, frequency, quartiles_dict = quartiles dict)
    """
    if value <= quartiles_dict[param][0.25]:
        return 4
    elif value <= quartiles_dict[param][0.50]:
        return 3
    elif value <= quartiles_dict[param][0.75]:
        return 2
    else:
        return 1


if __name__ == "__main__":
    rfm_segmentation = calculate_rfm()
    plot_distribution_over_r_f(rfm_segmentation)
    count_number_of_customers_for_each_segment(rfm_segmentation)
